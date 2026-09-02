# PnL explain module (equity options) — design

**Date:** 2026-09-02
**Branch:** `worktree-pnl-explain` (off `main` @ 7528618)
**Status:** approved design, pending implementation plan

## 1. Motivation

The riskmeasures refactor (spec 2026-09-01) gave the equity stack a full
greek stencil: delta, gamma, vega, vanna, volga, the dual-clock theta suite
with r/q/convexity/gamma_theta components, rho, dividend_rho, charm, color,
speed, zomma, vega_theta, dividend_volga, delta_q. Nothing yet turns those
sensitivities, or the engines behind them, into an explanation of a realized
PnL. Both backtests record daily greeks and daily PnL side by side but never
attribute one to the other, and nothing in the library can diff two
`PricingEnvironment` snapshots into risk-factor moves.

This module adds a comprehensive PnL explain for equity option products: a
greeks-based Taylor explain and a full-revaluation waterfall, sharing one
factor model, with lifecycle cash flows (KO, coupon, expiry, settlement)
explained as an explicit event term, aggregated to position and portfolio
level, and emitted as a daily series by both backtest engines.

## 2. Decisions taken in brainstorming

| Question | Decision |
|---|---|
| Attribution method | Both: Taylor (greeks × moves, with residual) and sequential full-revaluation waterfall (exact by construction). One `FactorDiff` feeds both so they reconcile row by row. |
| Product scope | Everything the `GreeksCalculator` handles (vanilla, American, barrier, digital, autocallables via PDE/QUAD/MC, delta-one hedges) for the market-factor explain, **including** lifecycle cash flows for every product the lifecycle trackers back (snowball, phoenix, the barrier family: KO / KI / coupon / maturity / expiry / settlement). Products without a tracker (vanilla, American, digital, delta-one) have no cashflow ledger in the library today; their expiry shows up as the engine's own intrinsic value at T = 0 and is explained as market moves, and a generic expiry / exercise / settlement transition for them is a follow-up (§15). |
| Layers | Instrument two-snapshot API, position/portfolio aggregation with trade and cost rows, daily explain series from both backtest engines. |
| Factor granularity | Scalar factors at the product coordinate by default; tenor-vega and key-rate rows opt-in via the bucketed-greeks machinery. |
| Architecture | New top-level `quantark/pnlexplain/` with an `equity/` subpackage (same shape as var / stresstest / dynamicscenario), an immutable `ValuationSnapshot`, a `FactorDiff` kernel, two explainers, a thin backtest recorder. |
| Display | Every Taylor row carries the raw derivative **and** the cash greek (vega per 1 vol point, rho per 1%), plus the moves in display units. PnL is always computed from raw derivatives. |

## 3. Reusable pieces already in the repo

- `GreeksCalculator` facade + `greeks/registry.py` (names, aliases, clock
  qualifiers, analytical auto-set), `greeks/numerical.py`,
  `greeks/analytical.py`, `greeks/theta_decomposition.py`
  (`estimate_theta_components`, `exact_theta_components`).
- `greeks/bump_envs.py`: `resolve_bump_engine` (frozen-seed bump context for
  MC → common random numbers), `resolve_theta_bump_mode`,
  `time_days_per_year`, `advance_theta_bump`.
- Bucketed coordinates (`VOL_TENOR_VEGA`, `RATE_KEYRATE`) with per-bump `pnl`
  fields.
- `param/vol/sticky.py::shocked_surface` (sticky-strike / sticky-moneyness
  views) and `GreekConvention`.
- `greek_conventions_report.py` cash conventions: `delta_cash = Δ·S`,
  `gamma_cash_1pct = Γ·S²/100`, `vega_1pct`, `theta_1d`, `rho_1pct`,
  `rhoq_1pct`.
- Lifecycle: `AutocallableLifecycleState` / `BarrierLifecycleState` with an
  append-only `LifecycleCashflowLedger` (`pending_pv(point, env)`,
  `paid_total(point)`), `ValuationPoint`, `LifecycleEvent`,
  `PortfolioLifecycleManager` (value identity: live MTM + pending receivable
  PV + paid cash), trackers' `product_for_pricing(date, env)`.
- Backtests: `backtest/equity/engine.py::BacktestEngine` (portfolio +
  lifecycle manager + hedge executor, `BacktestState` rows) and
  `backtest/replay/engine.py::ReplayBacktestEngine` (per-product
  `ProductReplay`, daily recalibrated vol-model engines, futures/spot hedge
  leg, `states`/`greeks` rows).

## 4. Package structure

```
quantark/pnlexplain/
├── __init__.py            # re-exports the public API
├── base.py                # Factor, ExplainMethod, ExplainRow, ValueBreakdown,
│                          #   PnLExplainResult
├── config.py              # PnLExplainConfig
├── README.md              # module guide (shipped; CLAUDE.md files are untracked)
└── equity/
    ├── __init__.py
    ├── snapshot.py        # ValuationSnapshot, value()
    ├── factor_diff.py     # FactorDiff, FactorMoves, change detection
    ├── waterfall.py       # WaterfallExplainer (sequential + shapley)
    ├── taylor.py          # TaylorExplainer, greek route/unit normalisation,
    │                      #   cash-greek display table
    ├── lifecycle.py       # alive-product handling, event row, receivable/paid cash
    ├── portfolio.py       # PositionSnapshot, BookSnapshot, explain_position,
    │                      #   explain_portfolio, trade / cost / hedge-leg rows
    └── recorder.py        # PnLExplainRecorder for both backtest engines
```

`example/pnl_explain_demo.py` demonstrates a vanilla, a barrier and a
snowball across a KO day.

## 5. Core types

### 5.1 `ValuationSnapshot` (frozen)

```python
@dataclass(frozen=True)
class ValuationSnapshot:
    product: BaseEquityProduct
    engine: BaseEngine
    pricing_env: PricingEnvironment
    date: datetime                      # must equal pricing_env.valuation_date
    quantity: float = 1.0
    lifecycle_state: Optional[EquityOptionLifecycleState] = None
    valuation_point: Optional[ValuationPoint] = None   # default: ValuationPoint(date=date)
    currency: Optional[str] = None      # label only; a book must be single-currency (§9)
    label: str = ""
```

Validation on construction: `date` is a naive, midnight `datetime` equal to
`pricing_env.valuation_date` (timezone-aware or intraday values raise;
fractional-day intervals are not supported); `spot` is positive and finite;
`quantity` is non-zero and finite; a date-based `valuation_point` must equal
`date`, and a numeric one is accepted as the caller's clock (§5.2 states the
consistency rule across two snapshots).

The module never mutates a snapshot's objects; every revaluation works on a
`deepcopy` of the environment. Callers whose loops mutate in place (both
backtests) pass copies, exactly as they do today for bump repricing. The
dataclass is frozen but the referenced product and engine are not: an engine
mutated **in place** between two snapshots is unsupported (both snapshots
would see the new state and `MODEL` would read unchanged). `MODEL` is
detected by engine identity, which is sufficient for the two backtests: the
replay creates fresh engine objects on every calibration day and the equity
backtest never recalibrates.

### 5.2 Value identity

```python
@dataclass(frozen=True)
class ValueBreakdown:
    contingent_mtm: float          # quantity × engine.price(product, env); 0 once terminal
    pending_receivable_pv: float   # ledger.pending_pv(valuation_point, env)
    paid_cash: float               # ledger.paid_total(valuation_point)
    total: float

def value(snapshot: ValuationSnapshot, *, engine=None) -> ValueBreakdown
```

Terminal means the lifecycle state reports `alive is False`, `knocked_out`,
`matured` or `expired` (the same predicate as
`settlement_support.terminal_lifecycle_pv`). Ledger amounts are already
quantity-scaled (the trackers register position-level cashflows), so only
the contingent leg is multiplied by `quantity`. This is the identity both
backtests already use for portfolio value, so the explain reconciles to
their daily PnL by construction.

**Unit contract.** `engine.price` is a per-unit value and `quantity` counts
those units: an option contract (the product's `contract_multiplier` is
inside the engine price), a share for `SpotInstrument`, and for `Futures`
one unit of the quoted price (the delta-one engine returns the futures
price per unit **without** the contract multiplier, and the hedge executor
books futures positions in those units). Trade prices (§9) are per unit in
the same convention; contract multipliers appear only where a recorder
converts a contract count into units.

**One scenario cache.** `pv_t0`, `pv_alive_t1` and `pv_t1` are read from
the same per-call scenario cache the waterfall rows use (`value(state({}))`,
`value(state(all))`, and the t1 snapshot valued on the same bump-context
engine), never recomputed separately, so the exactness claims of §6 hold
for MC engines as well.

Ledger entries carrying numeric-time (not date) payment representations
require the caller to supply a time-based `valuation_point`; the replay
recorder does (it knows the start date). The two snapshots' numeric points
must share one origin: `time_t1 − time_t0 == calendar_days / 365` within
1e-12 (the replay's own elapsed-time convention), otherwise
`ValidationError`. A missing representation raises `ValidationError`, never
silently assumes zero.

### 5.3 `FactorDiff` / `FactorMoves`

Built once from two snapshots (and the lifecycle transition, §8) and consumed
by both explainers.

```python
class Factor(Enum):
    TIME, SPOT, VOL, RATE, DIVIDEND, BASIS, MODEL, LIFECYCLE_EVENT,
    TRADE, TRANSACTION_COST, UNEXPLAINED, TOTAL

MARKET_FACTORS = (TIME, SPOT, VOL, RATE, DIVIDEND, BASIS, MODEL)

@dataclass(frozen=True)
class FactorCoordinate:
    reference_strike: Optional[float]   # K*: product.strike, else initial_price, else S0
    tenor_t1: Optional[float]           # T1 in years as seen at t1; None when no maturity
    applicable: frozenset[Factor]       # market factors this product's value can depend on

@dataclass(frozen=True)
class FactorMoves:
    coordinate: FactorCoordinate
    spot_t0: float; spot_t1: float; d_spot: float; spot_return: float
    vol_t0: Optional[float]; vol_t1: Optional[float]; d_vol: Optional[float]      # at (K*, T1)
    rate_t0: Optional[float]; rate_t1: Optional[float]; d_rate: Optional[float]   # at T1
    div_t0: Optional[float]; div_t1: Optional[float]; d_div: Optional[float]      # at T1
    basis_t0: Optional[float]; basis_t1: Optional[float]; d_basis: Optional[float]
    calendar_days: int
    trading_days: Optional[int]                         # None without a calendar
    year_fraction: float                                # env day count, t0 → t1
    changed: frozenset[Factor]
```

**Coordinate resolver.** `resolve_coordinate(product_t0, spot_t0,
product_alive_t1, env_t1) -> FactorCoordinate`:

- `reference_strike`: `product.strike` when the product has a positive
  strike, else `product.initial_price` when present, else `spot_t0` (the rule
  every vega bump uses, with the t0 spot pinned explicitly).
- `tenor_t1`: `product_alive_t1.get_maturity(env_t1)` for any product with a
  maturity (date-based or float), the expiry for `Futures`, `None` for a
  `SpotInstrument`.
- `applicable`: options → all seven market factors; engine-priced `Futures`
  → TIME, SPOT, RATE, DIVIDEND, BASIS, MODEL; `SpotInstrument` → SPOT,
  MODEL; the replay's quoted futures hedge leg (§9) → SPOT, BASIS. A
  non-applicable factor's moves are `None` and its rows are emitted as zero
  without pricing.
- Autocallables with several observation strikes still get one coordinate
  (initial-price strike, final maturity): the scalar vol move is a summary
  at that coordinate; the bucketed opt-in (§7.6) is the term-structure view.

**Sampling rules.** `vol_t*` = `env.get_vol(K*, T1)` (spot argument as the
surface requires), `rate_t*` = `env.get_rate(T1)`, `div_t*` =
`env.get_div_yield(T1)`, `basis_t*` = `env.get_basis_yield(T1)`; each
environment interpolates or extrapolates with its own objects' rules. On an
expiry day (`tenor_t1 <= 0`) VOL / RATE / DIVIDEND / BASIS are not
applicable and their moves are `None`. A `None` dividend or basis object is
normalised to the environment's zero-yield object before comparison, so
`None → ContinuousDividendYield(0.0)` is unchanged; any other structural
change (flat → term structure) is simply a changed object. Display units:
`spot_return = d_spot / spot_t0`, `vol_pts = d_vol × 100`, `rate_pct` /
`div_pct` / `basis_pct` = move × 100, `days = calendar_days`,
`trading_days` present only when a calendar exists. `moves` mappings omit
unavailable keys rather than carrying `None`.

Δσ, Δr, Δq are read **at T1 on both sides** deliberately: TIME is the first
step of the default order, so roll-down along the t0 term structure belongs
to `time` and the `vol` / `rate` / `dividend` rows measure the market move only.

**Factor ownership.** Each market factor owns exactly these inputs; the
waterfall swaps nothing else:

| Factor | Owns |
|---|---|
| TIME | `valuation_date`, the valuation point, and the contract roll (t0 product → alive-at-t1 product) |
| SPOT | `spot_quote` |
| VOL | `vol_surface` |
| RATE | `rate_curve` |
| DIVIDEND | `div_yield` |
| BASIS | `basis_yield` |
| MODEL | `engine` |
| LIFECYCLE_EVENT | `lifecycle_state` and the post-event product / engine substitution carried by the transition |

Every other `PricingEnvironment` field must compare equal between the
snapshots (`day_count_convention`, `bus_days_in_year`, calendar), otherwise
`ValidationError`: a clock change is not a market move. Calendar equality is
**semantic**: same class, same `name`, equal `holidays` sets. Independently
deep-copied calendars therefore compare equal; different calendars do not.

**Contract identity.** The alive-at-t1 product must be the t0 contract rolled
in time, checked by `contract_fingerprint(product)`: the product's class
name plus every dataclass field, with the **rolled fields** excluded and the
schedule-bearing fields reduced to their static terms. Rolled fields are the
ones the trackers legitimately change day to day: `maturity` (float),
`_otc_lifecycle_knocked_in`, and the observation *timing* inside
`barrier_config` / `post_barrier_config` / `observation_schedule` /
coupon schedules (their barrier levels, rates, counts and indices stay in
the fingerprint). Everything else — notional, `contract_multiplier`,
strikes, `initial_price`, coupon and rebate rates, option type, barrier
levels and types, exercise style, settlement convention, `maturity_date` —
must be equal. In addition the roll itself is checked: for float
maturities `maturity_alive_t1 == maturity_t0 − calendar_days / 365` within
1e-12 unless both are at the tracker's `1e-8` floor; date-based products
keep identical dates. Any mismatch raises `ValidationError("contract
replacement is not a time step")`. Contract replacements enter only as
lifecycle transitions (§8) or as disappearance / appearance of position ids
with trades (§9).

**Change detection.** A market object is unchanged when it is the same object
or compares equal (`==`; dataclass equality for the flat / term-structure
types; a comparison that raises counts as changed). Unchanged factors emit a
zero row without pricing. `TIME` is always changed (`date_t1 > date_t0` by
validation). `MODEL` is changed when `engine_alive_t1 is not engine_t0`.
`LIFECYCLE_EVENT` is changed when the transition's state fingerprints differ
(§8).

**Validation.** `date_t1 <= date_t0` → `ValidationError`.

### 5.4 `ExplainRow` / `PnLExplainResult`

```python
class ExplainMethod(Enum):
    WATERFALL = "waterfall"; TAYLOR = "taylor"; SHARED = "shared"

class RowKind(Enum):
    COMPONENT = "component"          # additive within its method
    INFORMATIONAL = "informational"  # sub-decomposition shown next to a component, never summed
    SUMMARY = "summary"              # total, never summed

@dataclass(frozen=True)
class ExplainRow:
    factor: Factor
    term: str                      # waterfall: factor.value; taylor: greek / sub-row name
    method: ExplainMethod
    kind: RowKind
    level: str                     # "instrument" | "position" | "portfolio"
    pnl: float                     # position-level (quantity-scaled)
    moves: Mapping[str, float]     # display units, e.g. {"spot_return": .012, "vol_pts": -.5}
    greek: Optional[float] = None  # raw derivative (taylor rows)
    cash_greek: Optional[float] = None
    step: Optional[int] = None     # sequential waterfall order index; None for shapley
    metadata: Mapping[str, Any] = MappingProxyType({})

@dataclass(frozen=True)
class PnLExplainResult:
    date_t0: datetime; date_t1: datetime
    pv_t0: ValueBreakdown; pv_alive_t1: ValueBreakdown; pv_t1: ValueBreakdown
    total_pnl: float               # pv_t1.total − pv_t0.total
    moves: FactorMoves
    rows: tuple[ExplainRow, ...]
    unexplained: Optional[float]   # taylor residual on the alive-contract move (a COMPONENT row too); None when Taylor is not requested
    metadata: Mapping[str, Any]

    def rows_for(self, method: ExplainMethod, *, kind: Optional[RowKind] = None) -> tuple[ExplainRow, ...]
    def by_factor(self, method: ExplainMethod) -> dict[str, float]   # COMPONENT rows only
    def reconcile(self, method: ExplainMethod) -> float             # total_pnl − Σ components(method ∪ shared)
    def to_frame(self) -> pd.DataFrame     # one row per ExplainRow, with kind and level columns
    def to_dict(self) -> dict
```

Rows are emitted in a fixed order (waterfall steps in order, then the event
row, then Taylor rows in stencil order with each informational sub-row
directly after its parent, then `unexplained`, then `total`); no path builds
a row set by iterating a set or dict of floats.

**Additivity contract.** For each requested method M, the sum of `pnl` over
rows with `kind == COMPONENT` and `method in {M, SHARED}` equals
`total_pnl`: exactly for the waterfall (by construction) and for Taylor
(because `unexplained` is a Taylor component). `INFORMATIONAL` rows (theta
sub-rows, `gamma_theta`, the key-rate parallel) and `SUMMARY` rows (`total`)
are never summed. `PnLExplainResult.reconcile(method) -> float` implements
the rule and returns the gap; `to_frame()` carries `kind` and `level`
columns, and frames that mix position and portfolio levels are reconciled
per level.

### 5.5 `PnLExplainConfig` (frozen)

```python
@dataclass(frozen=True)
class PnLExplainConfig:
    methods: tuple[ExplainMethod, ...] = (WATERFALL, TAYLOR)
    waterfall_order: tuple[Factor, ...] = (TIME, SPOT, VOL, RATE, DIVIDEND, BASIS, MODEL)
    interaction: str = "sequential"          # | "shapley"
    spot_convention: GreekConvention = GreekConvention.STICKY_STRIKE
    stencil: str | Sequence[str] = "standard"  # "first_order" | "standard" | "extended" | names
    time_term: str = "exact_gap"             # | "per_step"
    clock: Optional[str] = None              # per_step only: None (resolved) | "1d" | "1td"
    theta_decomposition_mode: str = "estimate"   # | "exact"
    bucketed: bool = False
    greeks_method: str = "auto"              # "auto" | "analytical" | "numerical"
    greeks_mode: GreeksCalculationMode = GreeksCalculationMode.BUMP
    params: Optional[EngineParams] = None    # bump sizes; default engine.params
```

Validation: `methods` is a non-empty, duplicate-free subset of `{WATERFALL,
TAYLOR}` (`SHARED` is never requestable); `waterfall_order` is a
permutation of all seven `MARKET_FACTORS` (zero rows are kept for unchanged
or non-applicable factors, so a sequential path always reaches the t1 state
and the exactness claim in §6 holds for every accepted order);
`interaction ∈ {"sequential", "shapley"}`, `time_term ∈ {"exact_gap",
"per_step"}`, `theta_decomposition_mode ∈ {"estimate", "exact"}`,
`greeks_method ∈ {"auto", "analytical", "numerical"}`, `clock ∈ {None,
"1d", "1td"}` and only with `per_step`; an explicit `stencil` list is
normalised through the greeks registry (aliases resolved, clock qualifiers
rejected because the explainer owns the clock, duplicates rejected, the
theta sub-rows `r_theta` / `q_theta` / `convexity_theta` / `gamma_theta`
accepted only together with `theta`). All violations raise
`ValidationError`.

### 5.6 Public API

```python
from quantark.pnlexplain import (
    # kernel
    ValuationSnapshot, ValueBreakdown, value, FactorCoordinate, FactorMoves,
    PnLExplainConfig, PnLExplainResult, ExplainRow, RowKind, Factor,
    MARKET_FACTORS, ExplainMethod, explain,
    # lifecycle
    LifecycleTransition, lifecycle_fingerprint, contract_fingerprint,
    # position / portfolio
    ExplainTrade, PositionSnapshot, BookSnapshot, PositionExplainResult,
    PortfolioExplainResult, explain_position, explain_portfolio,
    # backtest
    PnLExplainRecorder,
)

def explain(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    *,
    config: Optional[PnLExplainConfig] = None,
    transition: Optional[LifecycleTransition] = None,   # §8
) -> PnLExplainResult
```

Rows carry `level="instrument"` and money amounts for the whole position
(the contingent leg scaled by `snapshot_t0.quantity`, ledger amounts as
booked); the position layer relabels them (§9). A quantity change between
the snapshots is a trade and belongs to the position layer, so `explain`
raises `ValidationError` when the quantities differ.

## 6. Waterfall explainer

Sequential full revaluation of the **t0 contract** (t0 product / alive-at-t1
product, t0 lifecycle ledger), moving one factor at a time from the t0
environment toward t1. All pricings go through one bump-context engine
resolved once per snapshot (`resolve_bump_engine`), so MC engines price every
step under common random numbers.

Default order and what each step swaps:

| # | Factor | Swap | Notes |
|---|---|---|---|
| 1 | `time` | (product_t0, date_t0) → (product_alive_t1, date_t1); valuation point → t1 | Date-based products: same object, date advances. Tracker-rolled products: the tracker's pricing product for t1 built **before** that day's events (§8). A pending receivable's discount unwind and any payment falling in (t0, t1] land here. |
| 2 | `spot` | spot quote → S1 | Sticky-strike (default): t0 surface held. Sticky-moneyness: the vol component follows the precedence rule below, so the result is the same whichever of SPOT / VOL is applied first. |
| 3 | `vol` | vol surface → surface_t1 | LV/Heston/SLV engines price from their calibrated kernel, so this step is ~0 for them by construction; the move arrives in `model`. Rows stay separate and the README says so. |
| 4 | `rate` | rate curve → curve_t1 | |
| 5 | `dividend` | dividend yield → div_t1 | |
| 6 | `basis` | basis yield → basis_t1 | Futures-hedged books only; skipped when both are None. |
| 7 | `model` | engine → engine_t1 | Skipped without pricing when the engine object is identical. |
| — | `lifecycle_event` | (product_alive_t1, state_t0) → (product_t1, state_t1) at the t1 market | `pv_t1.total − pv_alive_t1.total`; zero when nothing fired. Events attached as metadata. Shared with Taylor (`method=SHARED`). |

**State construction.** A scenario state is a pure function of the *set* of
market factors applied so far, `state(S)`, never of the order they were
applied in: each factor in `S` sets the component it owns (§5.3) to its t1
value, every other component stays at t0, with one precedence rule for the
vol component under sticky-moneyness: `surface_t1` if `VOL ∈ S`, else
`shocked_surface(surface_t0, S0, S1)` if `SPOT ∈ S`, else `surface_t0`. The
sequential waterfall walks `S = {}`, `{f1}`, `{f1, f2}`, … along the
configured order and each step's row is `value(state(S_k)) −
value(state(S_{k−1}))`, where `value` prices the contingent leg on the
current engine and revalues the **t0 ledger** under the current
environment and valuation point (a receivable determined before t0 whose
payment date falls in `(t0, t1]` therefore accrues and pays inside TIME).
Because `state(all seven)` is the t1 market with the t0 lifecycle state, the
steps plus the event row sum to `total_pnl` exactly for every accepted
order; the waterfall has no residual row. Applying SPOT then VOL or VOL then
SPOT under sticky-moneyness reaches the same final state (tested).

**Pure time value.** `time_pure = value(state({TIME})) − pv_t0.total` is the
t0-market revaluation of the alive contract at t1, defined independently of
the waterfall order. It equals the sequential TIME row only when TIME is the
first step (the default order); the Taylor explainer (§7.4) always uses
`time_pure`.

`interaction="shapley"`: price `state(S)` for every subset `S` of the
changed, applicable market factors and allocate each factor its Shapley
value; rows carry `step=None`. 2ⁿ pricings; intended for analytical engines
and for validating a chosen order. The sum equals the sequential total.

Waterfall rows are `COMPONENT` rows carrying `moves` (§7.5 display units)
and metadata: `changed`, `engine` repr, and for the event row the fired
events from the transition (§8).

## 7. Taylor explainer

Greeks at t0 × factor moves, explaining the same alive-contract move as the
waterfall (`pv_alive_t1.total − pv_t0.total`). The `lifecycle_event` row is
shared, so Taylor never tries to explain a KO jump with gamma.

### 7.1 Greek route

The explainer never calls the facade with `method="auto"`. A new public
helper on the facade, `GreeksCalculator.resolve_route(product, greeks) ->
"analytical" | "numerical"`, exposes the existing routing rule (European
vanilla and every requested name in the analytical auto-set → analytical)
without changing any number; the explainer calls it and then requests
`calculate(..., method=route)`. `greeks_method="analytical"` /
`"numerical"` force the route (analytical on a non-vanilla raises as today).

### 7.2 Unit normalisation (reported → derivative)

| Greek | Reported convention | Derivative used |
|---|---|---|
| delta, gamma, speed, vanna, volga, zomma, dividend_volga, delta_q | raw derivatives | as is |
| vega | raw PnL per `vol_bump` (numerical) or per 1 vol pt (analytical) | vega / scale, scale = `vol_bump` or 0.01 |
| vega_theta | vega units per step | vega_theta / scale, per step |
| rho, dividend_rho | per +1% | value / 0.01 |
| theta, r_theta, q_theta, convexity_theta, gamma_theta, charm, color | per clock step | as is × steps |

Numerical one-sided vega carries a ½·volga·bump term; it lands in the
residual and is documented, not corrected.

### 7.3 Stencils and terms

`first_order` = delta, vega, theta, rho, dividend_rho.
`standard` (default) = first_order + gamma, volga, vanna.
`extended` = standard + speed, zomma, charm, color, vega_theta,
dividend_volga, delta_q. An explicit name list is accepted (registry names).

```
delta·ΔS + ½gamma·ΔS² + ⅙speed·ΔS³
vega'·Δσ + ½volga·Δσ² + vanna·ΔS·Δσ + ½zomma·ΔS²·Δσ
theta_gap (+ sub-rows) + charm·ΔS·n + ½color·ΔS²·n + vega_theta'·Δσ·n
rho'·Δr + dividend_rho'·Δq + ½dividend_volga·Δq² + delta_q·ΔS·Δq
unexplained = (pv_alive_t1 − pv_t0) − Σ terms
```

Greeks requested from the calculator in one call so the scenario memo dedupes
pricings; greeks not in the stencil are not requested.

**Term → factor.** Every Taylor row carries the factor of its highest-order
market input, cross terms included:

| Factor | Terms |
|---|---|
| TIME | theta (+ informational sub-rows), charm, color, vega_theta |
| SPOT | delta, gamma, speed |
| VOL | vega, volga, vanna, zomma |
| RATE | rho |
| DIVIDEND | dividend_rho, dividend_volga, delta_q |
| BASIS, MODEL | no Taylor rows; their effect is part of `unexplained` (documented) |

**Scaling.** `greek` and `cash_greek` are **position-level** (per-unit value
× `quantity`), so `pnl = coefficient × greek × moves` holds row by row with
the moves in raw units (`ΔS`, `Δσ`, `Δr`, `Δq`, `n`); the display `moves`
mapping carries the same moves in display units.

### 7.4 Time term

`time_term="exact_gap"` (default): the `theta` component row **is**
`time_pure` (§6, computed once and shared with the waterfall), so it equals
the sequential TIME row whenever TIME is the first step. Its `greek` and
`cash_greek` columns show the average per calendar day, `time_pure /
calendar_days`, with `moves = {"days": calendar_days, "trading_days": …}`;
the PnL is a full revaluation including ledger carry, not a derivative, and
the metadata says so (`"basis": "revaluation"`). Two theta bases are named
and kept apart:

- `time_pure` — the `theta` **component** row: alive contract **and** t0
  ledger revalued at t1 on the t0 market (§6).
- `theta_contract` — an **informational** sub-row: the calculator's theta
  of the contingent contract alone over the same calendar gap (no ledger).

Informational sub-rows under `theta`, never summed, never in the residual:
`theta_contract`; `ledger_carry := time_pure − theta_contract` (receivable
accretion and payments inside the gap; zero without a ledger); `r_theta`
and `q_theta` from the calculator's decomposition of `theta_contract`
(`theta_decomposition_mode` passed through; `estimate` components are per
day and are multiplied by `calendar_days`, `exact` reprices with the gap
step directly); `convexity_theta := theta_contract − r_theta − q_theta` by
definition in both modes; and under `extended` `gamma_theta`, with
`theta_residual := convexity_theta − gamma_theta` recorded in metadata.
Every equation above is stored verbatim in the row metadata (`"formula"`).

The other time-family greeks (charm, color, vega_theta) are measured **over
the gap step** in this mode, so their terms use `n = 1` — they are already
whole-gap quantities and are never multiplied by the day count again. Their
`greek` / `cash_greek` display columns are the per-day averages (gap value /
`calendar_days`) so the table reads in the same units as `per_step`.

Mechanically, the explainer builds one `GreeksCalculator` per call from
`config.params` (default `engine.params`) with the bump config overridden to
`time_bump_days = calendar_days`, `time_bump_mode = "calendar_days"`, so a
single `calculate()` request covers every stencil greek and the scenario
memo dedupes the pricings. Time-family greeks are evaluated on the alive
contract roll carried by the transition (the calculator's own date advance
is used for the bump scenario; for tracker-rolled products the gap-step
equality is guaranteed only for `theta`, which is `time_pure` by
definition, and the charm / color / vega_theta scenario difference is part
of the residual — covered by the multi-day lifecycle-product test). A
terminal-at-t0 position has no contingent leg: its Taylor rows are zero
except the shared `theta` row.

`time_term="per_step"`: per-day greeks on the resolved clock (`clock=None` →
`BumpConfig.time_bump_mode` resolution, `"1d"` / `"1td"` → qualified names)
× `n` steps (calendar days or trading days); the theta sub-rows are again
informational. This is the classic desk report; it misexplains a weekend or
an observation date and the difference goes to `unexplained`.

### 7.5 Display contract (raw + cash + moves)

PnL is computed from raw derivatives; the display columns cannot change a
number. Cash conventions follow `greek_conventions_report.py`:

| Term | Raw | Cash greek | Move keys (display units) |
|---|---|---|---|
| delta | dV/dS | Δ·S | `spot_return` |
| gamma | d²V/dS² | Γ·S²/100 | `spot_return` |
| speed | d³V/dS³ | ·S³/10⁴ | `spot_return` |
| vega | dV/dσ | ×0.01 (per 1 vol pt) | `vol_pts` (Δσ×100) |
| volga | d²V/dσ² | ×10⁻⁴ | `vol_pts` |
| vanna | d²V/dSdσ | ·S·0.01 | `spot_return`, `vol_pts` |
| zomma | d³V/dS²dσ | ·S²/100·0.01 | `spot_return`, `vol_pts` |
| theta family | per step | per day on the resolved clock | `days`, `trading_days` |
| charm | dΔ/dt | ·S per day | `spot_return`, `days` |
| color | dΓ/dt | ·S²/100 per day | `spot_return`, `days` |
| vega_theta | dvega/dt | ×0.01 per day | `vol_pts`, `days` |
| rho | dV/dr | ×0.01 (per 1%) | `rate_pct` (Δr×100) |
| dividend_rho | dV/dq | ×0.01 | `div_pct` |
| dividend_volga | d²V/dq² | ×10⁻⁴ | `div_pct` |
| delta_q | dΔ/dq | ·S·0.01 | `spot_return`, `div_pct` |

Waterfall rows carry the same move keys (no greek), so a report can show the
two methods side by side per factor. `S` in the cash columns is `spot_t0`.

### 7.6 Bucketed opt-in

Opt-in, delivered last (phase P5) behind `bucketed=True`; the default daily
explain never depends on it.

- **Canonical pillars** are the t0 object's pillars (`vol_surface.times`,
  `rate_curve.tenors`); the t1 object is sampled at those same pillars with
  its own interpolation / extrapolation, so differing t1 grids are allowed
  and the move at pillar τ is `σ1(K*, τ) − σ0(K*, τ)` (resp. `r1(τ) −
  r0(τ)`). A t0 object that is a term structure while the t1 object is not
  (or vice versa) raises `ValidationError` in bucketed mode.
- **Derivatives**: each `BucketedGreekPoint.derivative` from the existing
  machinery (already the signed-bump-normalised derivative, central or
  one-sided per the coordinate's difference mode) is multiplied by
  `quantity` and by the pillar move: `vega.<τ>` = `dV/dσ_τ × Δσ_τ`,
  `rho.<τ>` = `dV/dr_τ × Δr_τ`. The display columns follow §7.5 with the
  pillar in `moves["tenor"]`.
- **Rows**: bucket rows are the `COMPONENT` rows and the scalar `vega` /
  `rho` row they replace is not emitted, so nothing is counted twice; the
  key-rate parallel (`rate_keyrate.parallel × Δr_parallel`) is an
  `INFORMATIONAL` row with `sum_of_buckets` and `reconciles` in metadata.
  Reconciliation is unchanged: bucket rows sum into the Taylor components.
- Objects that are not term structures keep the scalar rows.

## 8. Lifecycle event term

```python
@dataclass(frozen=True)
class LifecycleTransition:
    product_alive_t1: BaseEquityProduct          # t0 contract rolled to t1, t0 state
    engine_alive_t1: BaseEngine                  # engine that prices it; target of MODEL
    state_before: Optional[EquityOptionLifecycleState]   # deep copy at t0
    state_after: Optional[EquityOptionLifecycleState]    # deep copy at t1
    events: tuple[LifecycleEvent, ...] = ()      # fired in (t0, t1], chronological

    @property
    def changed(self) -> bool: ...               # fingerprints differ (computed, never supplied)

def lifecycle_fingerprint(state) -> tuple        # versioned: ("v1", ...)
```

`product_alive_t1` is the contract as it would be priced at t1 had no event
fired in (t0, t1], carrying the **t0** lifecycle state. `engine_alive_t1` is
the engine that prices it and is the target of the `model` step. The
post-event product and engine are `snapshot_t1.product` /
`snapshot_t1.engine`; the transition never carries copies of them, it is
validated against them (below).

`lifecycle_fingerprint` is the semantic identity used for change detection
and is always **computed**, never accepted as an input. It is generic over
the state dataclasses so that a new pricing-relevant field can never be
missed: `("v1", type name, then every dataclass field of the state in
declaration order, normalised)` where a `LifecycleCashflowLedger` becomes
the tuple of its cashflows as `(cashflow_id, event_type.value, amount,
determination_date.isoformat() | determination_time, payment_date.isoformat()
| payment_time, sorted metadata items)`, a `ValuationPoint` becomes
`("date", iso)` or `("time", float)`, sets become sorted tuples, datetimes
become ISO strings, floats are kept exact. `None` states fingerprint to
`("v1", None)`. `LIFECYCLE_EVENT` is changed iff the two fingerprints
differ.

Rules (all violations raise `ValidationError`):

- `transition=None` is accepted only when the two snapshots' state
  fingerprints are equal; the kernel then builds the transition itself as
  (`snapshot_t1.product`, `snapshot_t1.engine`, equal states, no events).
- Fingerprints differ and `transition=None` → error (guessing the alive
  contract would silently mislabel the event row).
- A supplied transition is validated against the snapshots:
  `lifecycle_fingerprint(state_before) == fingerprint(snapshot_t0.state)`
  and `lifecycle_fingerprint(state_after) == fingerprint(snapshot_t1.state)`;
  `product_alive_t1` passes the contract-identity check against
  `snapshot_t0.product` (§5.3).
- **No event, no substitution**: when the fingerprints are equal the
  transition must carry no events, `snapshot_t1.engine is engine_alive_t1`,
  and `snapshot_t1.product` must pass contract identity against
  `product_alive_t1` — so an engine or product replacement can never be
  smuggled into the event row; it is `MODEL` (engine) or a trade (§9).
- When the fingerprints differ, `events` must be non-empty, chronological,
  with dates in `(t0, t1]`, and `snapshot_t1.product` / `engine` may differ
  from the alive pair (the substitution lands in the event row).
- Recorders build transitions from the one tracker method both backtests
  already use, `tracker.product_for_pricing(t1, env_t1)` (the replay's
  `product_for_date` is a thin wrapper around it), evaluated **before**
  `apply_lifecycle_events` / `process_day`, and attach the day's
  `LifecycleEvent`s afterwards.

An engine swap that accompanies a lifecycle event (the barrier tracker
replaces a knocked-in barrier with a vanilla **and** overrides the engine
with a `BlackScholesEngine`) belongs to the event row, not to `model`, so
recorders set `engine_alive_t1` to the engine that can price the *alive*
product: the replay recorder passes today's recalibrated engine (so
recalibration still lands in `model` on a KO day); the equity recorder
passes the position's engine as it was before lifecycle processing.

The event row is `pv_t1.total − pv_alive_t1.total` with `pv_alive_t1` valued
on (product_alive_t1, engine_alive_t1, env_t1, state_before, point t1), and
it covers **only what the transition changes**: newly determined cashflows,
terminal flags, product / engine substitution. On a KO day this is the jump
from the contingent claim to the fixed receivable; on a coupon day it is the
difference between the engine's own resolution of the t = 0 observation and
the tracker's determination (near zero when they agree); on an expiry day
the settlement. Payments of receivables determined **before** t0 are not
events: they accrue and pay inside TIME (§6). A position already terminal
at t0 has no contingent leg, so only `time` and `rate` can be non-zero and
its event row is zero.

## 9. Position and portfolio layers

```python
@dataclass(frozen=True)
class ExplainTrade:                      # the module's normalised trade schema
    position_id: str
    quantity: float                      # signed position UNITS (§5.2): buy > 0, sell < 0
    price: float                         # per unit, same convention as engine.price
    transaction_cost: float = 0.0        # >= 0, reduces PnL at the portfolio level
    timestamp: Optional[datetime] = None # in (t0, t1] when given
    kind: str = "adjust"                 # "open" | "adjust" | "close" | "roll_close" | "roll_open"
    instrument_type: str = ""
    metadata: Mapping[str, Any] = MappingProxyType({})
    # cash flow of the trade = -quantity * price (buying costs cash)

    @classmethod
    def from_contracts(cls, position_id, contracts, price, multiplier, **kw) -> "ExplainTrade":
        """contracts × multiplier units at the quoted price (futures recorders)."""

@dataclass(frozen=True)
class PositionSnapshot:
    position_id: str; underlying: str; snapshot: ValuationSnapshot
    tombstone: bool = False              # counterfactual / terminal snapshot for a position no longer in the book

@dataclass(frozen=True)
class BookSnapshot:
    date: datetime
    positions: Mapping[str, PositionSnapshot]        # live positions AND tombstones
    environments: Mapping[str, PricingEnvironment]   # per underlying, deep copies
    currency: Optional[str] = None                   # single currency per book; None = unlabeled
    @classmethod
    def from_portfolio(cls, portfolio, date, *, lifecycle_manager=None,
                       tombstones: Mapping[str, PositionSnapshot] = ...) -> "BookSnapshot"

def explain_position(
    pos_t0: Optional[PositionSnapshot],   # None only for a position opened in (t0, t1]
    pos_t1: PositionSnapshot,             # live, or a tombstone (see the case table)
    *,
    trades: Sequence[ExplainTrade] = (),
    transition: Optional[LifecycleTransition] = None,
    config: Optional[PnLExplainConfig] = None,
) -> PositionExplainResult

def explain_portfolio(
    book_t0: BookSnapshot,
    book_t1: BookSnapshot,
    *,
    trades: Sequence[ExplainTrade] = (),
    transaction_costs: float = 0.0,                   # costs not attached to a trade
    transitions: Mapping[str, LifecycleTransition] = MappingProxyType({}),  # by position_id
    config: Optional[PnLExplainConfig] = None,
) -> PortfolioExplainResult

@dataclass(frozen=True)
class PositionExplainResult:
    position_id: str; underlying: str
    instrument: Optional[PnLExplainResult]   # market + event rows at q0; None for a position opened today
    trade_rows: tuple[ExplainRow, ...]
    total_pnl: float                          # V1 − V0 + Σ trade cash, GROSS of transaction costs
    rows: tuple[ExplainRow, ...]              # level="position": relabeled instrument rows (its `total` dropped) + trade rows + one position `total`

@dataclass(frozen=True)
class PortfolioExplainResult:
    date_t0: datetime; date_t1: datetime
    positions: Mapping[str, PositionExplainResult]    # insertion order = sorted position_id
    rows: tuple[ExplainRow, ...]          # level="portfolio" rows (below) + cost row + one portfolio `total`
    total_pnl: float                      # Σ position totals − costs
    metadata: Mapping[str, Any]           # includes "reconciliation"
    def reconcile(self, method: ExplainMethod) -> float
    def by_underlying(self) -> Mapping[str, tuple[ExplainRow, ...]]
    def to_frame(self) -> pd.DataFrame
```

`from_portfolio` deep-copies the environment per underlying and each
position's lifecycle state; products and engines are referenced (the equity
backtest mutates them only through lifecycle substitution, which the
transition records). Both backtest recorders map their native trade records
onto `ExplainTrade` (§10); nothing in this layer reads `TradeRecord`
directly. A book is single-currency by contract: snapshots that carry a
`currency` label must all agree with the book's, otherwise `ValidationError`
(no FX conversion; multi-currency books are out of scope, §15).

**Level promotion.** `explain()` emits `level="instrument"` rows.
`explain_position` copies them relabeled to `level="position"`, drops the
instrument `total` summary, appends the trade rows and exactly one position
`total` (= `total_pnl`). `explain_portfolio` keeps every position's rows
and adds `level="portfolio"` rows plus exactly one portfolio `total`.

**Algebra (position values, units of §5.2).** `V` is
`ValueBreakdown.total`, `q` the position quantity in units, `u = V / q` the
per-unit value. A position's day PnL is `V1 − V0 + Σ_i(−q_i·p_i)`; trades
must reconcile, `Σ_i q_i == q1 − q0` within `1e-9 × max(1, |q1|, |q0|)`
(else `ValidationError`). Market and event rows are the instrument explain
at `q0`; the **trade** row for trade *i* is `q_i · (u1 − p_i)` (execution vs
the t1 model mark), so `market rows + event row + trade rows == V1 − V0 +
trade cash` holds exactly. Trade validation: finite, non-zero `quantity`;
finite `price`; `transaction_cost >= 0`; `kind` in the five names;
`timestamp`, when given, in `(t0, t1]`. Cases:

| Case | Inputs | Rows |
|---|---|---|
| unchanged quantity | both sides live | instrument rows only |
| quantity changed, no lifecycle state | both sides live + trades | instrument rows on the t1 snapshot rescaled to `q0` (no ledger, so the contingent leg is linear in `q`), plus one trade row per trade |
| quantity changed, lifecycle state present | — | `ValidationError`: the ledger is booked at the tracker's registered quantity and neither backtest trades a tracked position |
| opened today | `pos_t0=None`, live `pos_t1`, trades with `Σ q_i == q1` | trade rows only; `instrument=None`; `total = V1 + trade cash` |
| closed by trading | live `pos_t0`, **tombstone** `pos_t1` = counterfactual alive snapshot (alive product from the transition or the t0 product, `engine_alive_t1`, `env_t1`, quantity `q0`), closing trades with `Σ q_i == −q0` | instrument rows from the counterfactual; `u1` is its model mark; trade rows are the execution slippage; `total = −V0 + trade cash` |
| removed by lifecycle termination | live `pos_t0`, **tombstone** `pos_t1` = terminal snapshot (t0 product, engine, `state_after`, `env_t1`, `q0`, its valuation point), transition with terminal `state_after`, no trades | instrument rows incl. the event row; the contingent leg is zero and the value is receivable PV + paid cash |
| terminal on both sides | tombstone on both sides while the ledger has a pending cashflow | instrument rows: `time` and `rate` only, event row zero |
| rolled hedge | position ids are **contract-specific**: the old id closes (`roll_close` trades, closed-by-trading case) and the new id opens (`roll_open` trades, opened-today case); a roll is never netted into one trade and never reuses an id |
| absent at t1 with no tombstone | — | `ValidationError` |
| present at t1, absent at t0, no trades | — | `ValidationError` |

Recorders keep a tombstone for every lifecycle-terminated position until
its ledger has no pending cashflow (rolling its valuation point forward
each day) and for every position closed by trading on the day it closes.

- **transaction_cost**: one portfolio-level `COMPONENT` / `SHARED` row with
  `pnl = −(Σ trade.transaction_cost + transaction_costs)`; position results
  are gross of costs.
- **hedge legs** (replay engine): the futures hedge is a **quoted futures
  leg** — its value is `units × F` with `F` the market futures price, not
  an engine price — represented as a `Futures` position whose engine
  returns the quoted price (`use_market_price=True`, `market_price = F`).
  Its coordinate is `applicable = {SPOT, BASIS}` and its rows are `spot` =
  `units × ΔS` and `basis` = `units × (ΔF − ΔS)` (the futures curve's
  carry and basis are not separately identifiable from a quoted price); rolls
  are contract-specific ids with `roll_close` / `roll_open` legs built by
  `ExplainTrade.from_contracts`. Engine-priced `Futures` positions (equity
  backtest) go through the ordinary waterfall with the §5.3 applicable set;
  the two conventions never mix within one position.

**Aggregation.** `PortfolioExplainResult.rows` are aggregated by
(`underlying`, method, kind, factor, term) at `level="portfolio"`, with the
display columns `moves`, `greek`, `cash_greek` and `step` carried **only**
when every constituent position shares the same coordinate (underlying,
`reference_strike`, `tenor_t1`) and step, and null otherwise; `metadata =
{"positions": (ids…), "underlying": …}`. Book-level rows across underlyings
are aggregated by (method, kind, factor, term) with null display columns;
`by_underlying()` exposes the per-underlying tier. `to_frame()` has one row
per (position_id or `"portfolio"`, underlying or `"*"`, level, method, kind,
factor, term).

**Reconciliation** is per method and per level: `expected = Σ_positions
(V1 − V0 + trade cash) − costs`, `explained = Σ COMPONENT rows at that level
with method ∈ {M, SHARED}`; `metadata["reconciliation"][M] = {"expected",
"explained", "gap", "ok"}` with `ok = is_close(gap, 0, abs_tol = 1e-8 ×
max(1, |expected|))`, surfaced, never swallowed.

## 10. Backtest integration

Both configs gain `pnl_explain: Optional[PnLExplainConfig] = None`. When
None nothing changes: no extra pricings, `states` / `greeks` frames
byte-identical, goldens untouched.

`PnLExplainRecorder` keeps yesterday's `BookSnapshot` (or per-replay
snapshots) and the day's trade records, and emits rows:

- **Equity `BacktestEngine`** (`_step`): after `_update_pricing_environment`
  and before `_process_lifecycle`, build the day's transitions. The
  lifecycle manager substitutes `position.product =
  tracker.product_for_pricing(date, env)` every day (and the engine on a
  barrier KI), so the alive product must come from the tracker with the
  pre-event state: `PortfolioLifecycleManager` gains a pure accessor
  `pricing_products(portfolio, date) -> Dict[position_id, product]` (tracked
  positions → `tracker.product_for_pricing(date, env)`, untracked → the
  position's current product; no mutation). The recorder pairs each with the
  position's engine before `process_day` (`engine_alive_t1`) and a deep copy
  of the pre-event state; after `process_day` it attaches the returned
  `ProcessedLifecycleEvent.event`s and the post-event state copy. After
  hedging it builds today's `BookSnapshot` (with tombstones for positions
  the lifecycle manager removed today and for hedge contracts closed
  today), maps the day's `TradeRecord`s to `ExplainTrade` (quantity in
  units with the recorded sign, buy positive; `price`; `transaction_cost`;
  `kind` from `trade_type`), calls `explain_portfolio`, then stores today's
  snapshot as tomorrow's t0. Hedge positions are ordinary delta-one
  positions through the same layer; the executor books each hedge contract
  under its own position id, so rolls are close / open legs on distinct ids
  and executor `realized_pnl` is reproduced by them.
- **`ReplayBacktestEngine`** (`run` loop): per replay, call
  `tracker.product_for_pricing(date, env)` (via `product_for_date`) before
  `apply_lifecycle_events` to build the transition's alive product with the
  pre-event state, and read the post-event product / state afterwards; keep
  yesterday's engine reference so the daily recalibrated engine yields a
  `model` row (`engine_alive_t1` = today's engine); map `_trades` rows and
  roll legs to `ExplainTrade.from_contracts` with the futures multiplier;
  explain the futures hedge as the quoted futures leg of §9 under the
  active contract's id (spot hedges as a `SpotInstrument`). The valuation
  point for numeric-time ledgers is built from the replay start date.

**Initialisation.** The first replay date has no t0: no explain rows are
emitted for it and `explain_df` starts on the second date, whose expected
PnL is the change in `total_pnl` between the first two state rows. Opening
positions on day one are simply the day-two t0 book. Restart / checkpoint
of a backtest is not a feature of either engine and none is added.

**Isolation.** All explain pricings run on `resolve_bump_engine` contexts
(fresh, seed-frozen for MC), so they neither advance a production engine's
RNG nor touch its caches; test 6 asserts that turning the explain on leaves
the `states`, `greeks` and `trades` frames byte-identical to the explain-off
run.

Rows land in `explain_df` on `BacktestResults`, `BookBacktestResults` and
`AutocallableBacktestResults` with the fixed column order `date, level,
position_id, underlying, method, kind, factor, term, step, pnl, greek,
cash_greek, spot_return, vol_pts, rate_pct, div_pct, basis_pct, days,
trading_days, tenor` (missing moves are NaN; enums serialised as `.value`;
`date` is `datetime64[ns]`), sorted by (`date`, level rank instrument <
position < portfolio, `position_id`, method order waterfall < taylor <
shared, `step`, stencil order). Both position-level and portfolio-level
rows are emitted. A companion `explain_reconciliation_df` has one row per
(`date`, `method`, `level`) with `expected` (that day's change in the states
frame's total PnL for the portfolio level; the position's own `total_pnl`
for the position level), `explained` (Σ COMPONENT rows with method ∈ {M,
SHARED} at that level), `gap` and `ok`. An empty result yields empty frames
with exactly these columns.

## 11. Error handling

| Condition | Behaviour |
|---|---|
| `snapshot.date != pricing_env.valuation_date` | `ValidationError` |
| `date_t1 <= date_t0` | `ValidationError` |
| invalid `PnLExplainConfig` (non-market factor in the order, duplicates, unknown stencil, `clock` with `exact_gap`) | `ValidationError` |
| clock mismatch between environments (day count, bus days, semantically different calendars) | `ValidationError` |
| contract identity check fails (alive product is not the t0 contract rolled in time) | `ValidationError` |
| lifecycle fingerprints differ, no `transition`; or a transition whose fingerprints do not match the snapshots | `ValidationError` |
| quantities differ in instrument-level `explain` | `ValidationError` |
| trades do not reconcile (`Σ q_i ≠ q1 − q0`), invalid trade fields (zero / non-finite quantity, non-finite price, negative cost, unknown kind, timestamp outside `(t0, t1]`), or trades on a lifecycle-tracked / lifecycle-terminated position | `ValidationError` |
| position gone at t1 with no tombstone; new at t1 without trades; a tombstone with trades that is terminal, or without trades that is not | `ValidationError` |
| transition without events while fingerprints differ, events outside `(t0, t1]` or out of order, or a product / engine substitution with equal fingerprints | `ValidationError` |
| currency labels disagree within a book | `ValidationError` |
| bucketed mode with a term-structure object on one side only | `ValidationError` |
| `"1td"` without `pricing_env.calendar` | the same `ValidationError` theta raises |
| numeric-time ledger without a time-based valuation point | `ValidationError` |
| non-finite step PnL | `NumericalError` naming the step |
| engine without `create_bump_context` | plain engine; documented (MC without CRN shows noise in the Taylor residual; the waterfall still sums exactly) |

No fallbacks that invent semantics: where the correct alive product, trade
price or valuation point is unknown, the module raises.

## 12. Testing

1. **Vanilla oracle** (`BlackScholesEngine`): waterfall steps + event row sum
   to `total_pnl` to machine precision; Taylor `standard` residual equals the
   closed-form Taylor remainder on a (ΔS, Δσ, Δr, Δt) grid within FD
   tolerance and `extended` shrinks it; `reconcile(M)` is zero for each
   method and summing every row (informational and summary included) is
   **not** the total, which the additivity test asserts explicitly.
2. **Shapley and order**: three changed factors; the Shapley allocation
   equals the brute-force average over all 6 orderings and sums to the
   sequential total; a reordered sequential waterfall still sums exactly
   while its TIME row differs from `time_pure`; under sticky-moneyness
   SPOT→VOL and VOL→SPOT reach the same final state; a non-permutation
   order raises.
3. **Time-step equivalence and contract identity**: a date-based and a
   float-maturity build of the same contract give the same `time` row; a
   different strike, or a replaced product without a transition, raises.
3b. **Coordinates and calendars**: `SpotInstrument` and `Futures` positions
   get zero vol / dividend rows without pricing and non-null `spot` /
   `basis` rows; deep-copied equal calendars are accepted and a
   `CHINA_SSE` vs `US` pair is rejected.
4. **Route and unit guard**: `BumpConfig(vol_bump=0.02)`, analytical and
   numerical routes agree with dV/dσ×Δσ within FD tolerance; cash columns
   equal `build_cash_greeks_report` values for the base greeks.
5. **Lifecycle days**: snowball KO day and Phoenix coupon day on quick PDE
   and QUAD configs; event row equals `pv_t1 − pv_alive_t1`; a terminal-at-t0
   position explains its receivable carry only and its event row is zero;
   a receivable determined before t0 and paid inside (t0, t1] moves from
   pending PV to paid cash inside TIME with a zero event row; a barrier KI
   day (product and engine substituted) puts the engine swap in the event
   row and leaves `model` at zero; a Friday→Monday gap on a tracker-rolled
   snowball gives `theta == time_pure` and charm / color rows whose PnL
   equals their gap value (no day-count double scaling).
5b. **Position cases** (§9 table): opened today, quantity adjusted,
   fully closed by trading, removed by lifecycle termination, and a rolled
   futures hedge with close / open legs each reconcile exactly;
   non-reconciling trades, trades on a tracked position, and an
   unexplained disappearance raise.
6. **Backtest reconciliation and isolation gate**: both engines on a short
   window with the explain on; `explain_reconciliation_df.ok` is true on
   every (date, method, level) with `abs_tol = 1e-8 × max(1, |expected|)`
   (spot delta hedge, multi-instrument option hedge with a roll, replay
   futures hedge with a roll, a lifecycle KO with a delayed settlement so a
   tombstone spans several days); the `states` / `greeks` / `trades` frames
   with the explain **on** are byte-identical to the explain-**off** run,
   and the explain-off run is byte-identical to `main` (same-machine
   invariant script, like the greeks fingerprint gate; not a committed
   golden).
6b. **MC common random numbers**: a snowball MC engine with a fixed seed;
   the waterfall reconciles exactly (gap ≤ 1e-10 × |total|) because every
   endpoint and step comes from one seed-frozen bump context, and the Taylor
   residual is reported, not asserted small.
7. **Clock**: Friday→Monday on a calendar env gives `days=3`, `trading_days=1`;
   `per_step` with `"1td"` uses one step, with `"1d"` three.
8. **Vol-model engine**: LV quick config across a recalibration; `model` row
   carries the surface move and `vol` is near zero.
9. **Bucketed**: term-structure surface and interpolated curve; bucket rows
   sum to the scalar rows within FD tolerance.
10. **Row order and frame schema**: `to_frame()` column and row order stable
    under several hash seeds (mirrors `test_greeks_key_order.py`); the
    empty frame has the §10 columns; Shapley rows, bucket pillars and
    same-timestamp trades keep their canonical sort.

**Tolerances used above.** "Machine precision" / "exactly" = `abs ≤ 1e-10 ×
max(1, |total_pnl|)`. "FD tolerance" between an analytical and a numerical
Taylor term = `rel ≤ 2e-2` or `abs ≤ 1e-6` (one-sided vega and the
second-difference terms carry O(bump) truncation). "Near zero" for the LV
`vol` row = `abs ≤ 1e-8 × max(1, |total_pnl|)`. "Shrinks" = the `extended`
residual is strictly smaller in absolute value than the `standard` residual
at every grid point whose spot or vol move is non-zero.

**Support matrix** (product family × engine family × method), each cell
covered by at least one acceptance case:

| Family | Engines | Waterfall | Taylor | Lifecycle rows |
|---|---|---|---|---|
| European vanilla | analytical, MC, PDE | ✓ (test 1) | ✓ analytical route (test 1, 4) | none (no tracker) |
| American | analytical BS93, PDE | ✓ | ✓ numerical route | none |
| Digital, Asian | analytical, MC | ✓ | ✓ numerical | none |
| Barrier family (barrier, one-touch, sharkfin) | analytical, PDE | ✓ | ✓ numerical | ✓ KI substitution, KO / expiry (test 5) |
| Autocallables (snowball, phoenix, KO-reset) | PDE, QUAD, MC | ✓ | ✓ numerical | ✓ KO / coupon / maturity / settlement (tests 5, 6, 6b) |
| Vol-model engines (LV, Heston, SLV) | PDE, MC | ✓ with `model` row (test 8) | ✓ numerical | as the wrapped product |
| Delta-one (`SpotInstrument`, engine-priced `Futures`) | delta-one analytical | ✓ (test 3b) | ✓ (delta only) | none |
| Quoted futures hedge leg (replay) | none (market price) | ✓ spot + basis | n/a | none |

Unsupported: products the `GreeksCalculator` cannot bump (raise from the
calculator, propagated), engines mutated in place (§5.1), multi-currency
books (§15).

## 13. Sequencing and gates

| Phase | Content | Gate |
|---|---|---|
| P0 | package skeleton, `ValuationSnapshot`, `value`, coordinate resolver, `FactorDiff` (ownership, contract identity, semantic calendar equality), waterfall (state-set construction, sequential + shapley, `time_pure`), row kinds and result types | tests 1–3, 3b, 10 |
| P1 | Taylor explainer, `resolve_route` helper, unit normalisation, display table, term → factor table, time term modes (gap scaling, named theta bases) | tests 4, 7; riskmeasures suite green and its fingerprint byte-identical |
| P2 | `LifecycleTransition` + computed fingerprint, event term, `ExplainTrade`, tombstones, position / portfolio layers incl. the §9 case table, aggregation tiers, cost / hedge rows | tests 5, 5b |
| P3 | recorders in both backtest engines, config fields, `explain_df` + `explain_reconciliation_df`, isolation | tests 6, 6b, 8; backtest and replay suites green |
| P4 | README, root-guide row, demo script, support-matrix acceptance cases not yet covered | full suite |
| P5 | bucketed opt-in (§7.6) — last, experimental, never on the default path | test 9 |

Every phase runs with worktree source shadowing the editable install
(`PYTHONPATH=$PWD`).

## 14. Compatibility contract

- No numeric change anywhere in `riskmeasures`: the only addition is the
  pure `resolve_route` helper. `PortfolioLifecycleManager` gains only the
  pure `pricing_products` accessor; `process_day` is untouched.
- Backtests with `pnl_explain=None` are unchanged in behaviour, cost and
  output.
- `PricingEnvironment`, products, engines and lifecycle types are untouched.

## 15. Out of scope

- Other asset classes (the package shape is ready for `fx/` etc.).
- Dynamic-scenario integration (same recorder pattern; separate follow-up).
- A generic expiry / exercise / settlement transition and cashflow ledger
  for products without a lifecycle tracker (vanilla, American, digital,
  delta-one): today they have no ledger anywhere in the library, so their
  expiry is explained as market moves (§2).
- Multi-currency books and FX conversion (a book is single-currency).
- Engines mutated in place between snapshots (§5.1).
- Backtest restart / checkpoint semantics (neither engine has them).
- Sticky-delta spot convention (rejected upstream in `sticky.py`).
- Model-parameter attribution inside a vol model (per-Heston-parameter rows);
  `model` is one row.
- A report / HTML renderer beyond `to_frame()`.
