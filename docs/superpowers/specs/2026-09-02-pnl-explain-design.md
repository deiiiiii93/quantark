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
| Product scope | Everything the `GreeksCalculator` handles (vanilla, American, barrier, digital, autocallables via PDE/QUAD/MC, delta-one hedges) **including** lifecycle cash flows. |
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
    label: str = ""
```

The module never mutates a snapshot's objects; every revaluation works on a
`deepcopy` of the environment. Callers whose loops mutate in place (both
backtests) pass copies, exactly as they do today for bump repricing.

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

Ledger entries carrying numeric-time (not date) payment representations
require the caller to supply a time-based `valuation_point`; the replay
recorder does (it knows the start date). A missing representation raises
`ValidationError`, never silently assumes zero.

### 5.3 `FactorDiff` / `FactorMoves`

Built once from two snapshots (and the alive-at-t1 product, §8) and consumed
by both explainers.

```python
class Factor(Enum):
    TIME, SPOT, VOL, RATE, DIVIDEND, BASIS, MODEL, LIFECYCLE_EVENT,
    TRADE, TRANSACTION_COST, UNEXPLAINED, TOTAL

@dataclass(frozen=True)
class FactorMoves:
    spot_t0: float; spot_t1: float; d_spot: float; spot_return: float
    vol_t0: float; vol_t1: float; d_vol: float          # absolute, at (K*, T1)
    vol_coordinate: tuple[float, float]                 # (K*, T1)
    rate_t0: float; rate_t1: float; d_rate: float       # at T1
    div_t0: float; div_t1: float; d_div: float          # at T1
    basis_t0: float; basis_t1: float; d_basis: float    # at T1; 0.0 when both None
    calendar_days: int
    trading_days: Optional[int]                         # None without a calendar
    year_fraction: float                                # env day count, t0 → t1
    changed: frozenset[Factor]
```

- `K*` is `product.strike` when the product has one, else spot — the same
  coordinate every vega bump uses. `T1` is the product maturity as seen at
  t1. Δσ, Δr, Δq are read **at T1 on both sides** deliberately: the time step
  is applied first, so roll-down along the t0 term structure belongs to
  `time` and the `vol` / `rate` / `dividend` rows measure the market move only.
- Change detection per factor: an object is unchanged when it is the same
  object or compares equal (`==`, dataclass equality for the flat/term
  structure types; a comparison that raises counts as changed). Unchanged
  factors emit a zero row without pricing. `TIME` is changed whenever
  `date_t1 > date_t0` or the product object differs; `MODEL` when
  `engine_t1 is not engine_t0`; `LIFECYCLE_EVENT` when the lifecycle states
  differ (ledger length, terminal flags, KI flag) or the t1 product differs
  from the alive-at-t1 product.
- Validation: `date_t1 <= date_t0` → `ValidationError`. Differing
  `day_count_convention`, `bus_days_in_year` or calendar identity between the
  two environments → `ValidationError` (a clock change is not a market move).

### 5.4 `ExplainRow` / `PnLExplainResult`

```python
class ExplainMethod(Enum):
    WATERFALL = "waterfall"; TAYLOR = "taylor"; SHARED = "shared"

@dataclass(frozen=True)
class ExplainRow:
    factor: Factor
    term: str                      # waterfall: factor.value; taylor: greek / sub-row name
    method: ExplainMethod
    pnl: float                     # position-level (quantity-scaled)
    moves: Mapping[str, float]     # display units, e.g. {"spot_return": .012, "vol_pts": -.5}
    greek: Optional[float] = None  # raw derivative (taylor rows)
    cash_greek: Optional[float] = None
    step: Optional[int] = None     # waterfall order index
    metadata: Mapping[str, Any] = MappingProxyType({})

@dataclass(frozen=True)
class PnLExplainResult:
    date_t0: datetime; date_t1: datetime
    pv_t0: ValueBreakdown; pv_alive_t1: ValueBreakdown; pv_t1: ValueBreakdown
    total_pnl: float               # pv_t1.total − pv_t0.total
    moves: FactorMoves
    rows: tuple[ExplainRow, ...]
    unexplained: float             # taylor residual on the alive-contract move
    metadata: Mapping[str, Any]

    def rows_for(self, method: ExplainMethod) -> tuple[ExplainRow, ...]
    def by_factor(self, method: ExplainMethod) -> dict[str, float]
    def to_frame(self) -> pd.DataFrame     # one row per ExplainRow
    def to_dict(self) -> dict
```

Rows are emitted in a fixed order (waterfall steps in order, then the event
row, then Taylor rows in stencil order, then `unexplained`, then `total`);
no path builds a row set by iterating a set or dict of floats.

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

Validation rejects an order containing `LIFECYCLE_EVENT`, `TRADE`,
`TRANSACTION_COST`, `UNEXPLAINED` or `TOTAL` (those are not market steps),
duplicates, unknown stencil names, and `clock` set together with
`time_term="exact_gap"`.

### 5.6 Public API

```python
from quantark.pnlexplain import (
    ValuationSnapshot, PnLExplainConfig, PnLExplainResult, ExplainRow, Factor,
    ExplainMethod, explain, explain_position, explain_portfolio,
    PositionSnapshot, BookSnapshot, PnLExplainRecorder,
)

def explain(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    *,
    config: Optional[PnLExplainConfig] = None,
    product_alive_t1: Optional[BaseEquityProduct] = None,
) -> PnLExplainResult
```

Rows are scaled by `snapshot_t0.quantity`; a quantity change between the
snapshots is a trade and belongs to the position layer (§9), so `explain`
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
| 2 | `spot` | spot quote → S1 | Sticky-strike (default): t0 surface held. Sticky-moneyness: `shocked_surface(surface_t0, S0, S1)` view is installed; the later `vol` step measures against that view. |
| 3 | `vol` | vol surface → surface_t1 | LV/Heston/SLV engines price from their calibrated kernel, so this step is ~0 for them by construction; the move arrives in `model`. Rows stay separate and the README says so. |
| 4 | `rate` | rate curve → curve_t1 | |
| 5 | `dividend` | dividend yield → div_t1 | |
| 6 | `basis` | basis yield → basis_t1 | Futures-hedged books only; skipped when both are None. |
| 7 | `model` | engine → engine_t1 | Skipped without pricing when the engine object is identical. |
| — | `lifecycle_event` | (product_alive_t1, state_t0) → (product_t1, state_t1) at the t1 market | `pv_t1.total − pv_alive_t1.total`; zero when nothing fired. Events attached as metadata. Shared with Taylor (`method=SHARED`). |

Each step's row is `value(state) − value(previous state)`, where `value`
prices the contingent leg and revalues the t0 ledger under the current
state's environment and valuation point. The steps plus the event row sum to
`total_pnl` exactly; the waterfall has no residual row.

`interaction="shapley"`: over the changed market factors, price the t0
contract under every subset (a subset's state is order-independent: each
factor sets one component; under sticky-moneyness the vol component is
`surface_t1` if `VOL` is in the subset, else the re-anchored view if `SPOT`
is, else `surface_t0`) and allocate each factor its Shapley value. 2ⁿ
pricings; intended for analytical engines and for validating a chosen order.
The sum equals the sequential total.

Waterfall rows carry `moves` (§7.5 display units) and metadata:
`changed`, `engine` repr, and for the event row the fired events.

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

### 7.4 Time term

`time_term="exact_gap"` (default): the `theta` row **is** the kernel's time
step (the waterfall step 1 value, computed once and reused), so it equals the
waterfall time row exactly. Sub-rows: `r_theta` and `q_theta` from the
calculator's decomposition over the same calendar gap
(`theta_decomposition_mode` passed through), and `convexity_theta := theta −
r_theta − q_theta` by definition in both modes. In `estimate` mode the
components come back per day and are multiplied by `calendar_days`; in
`exact` mode the zeroed-r/q repricings use the gap step directly. With
`extended`, `gamma_theta` (per day × `calendar_days`) is an informational
sub-row and `theta_residual = convexity_theta − gamma_theta` is recorded in
metadata. charm / color / vega_theta are likewise measured over the gap step.

Mechanically, the explainer builds one `GreeksCalculator` per call from
`config.params` (default `engine.params`) with the bump config overridden to
`time_bump_days = calendar_days`, `time_bump_mode = "calendar_days"`, so a
single `calculate()` request covers every stencil greek and the scenario
memo dedupes the pricings. A terminal-at-t0 position has no contingent leg:
its Taylor rows are zero except the shared time row.

`time_term="per_step"`: per-day greeks on the resolved clock (`clock=None` →
`BumpConfig.time_bump_mode` resolution, `"1d"` / `"1td"` → qualified names)
× the number of steps (calendar days or trading days). This is the classic
desk report; it misexplains a weekend or an observation date and the
difference goes to `unexplained`.

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

With `bucketed=True`: if the t0 vol surface is a `TermStructureVolSurface`,
the scalar `vega` row is replaced by one `vega.<tenor>` row per pillar
(`VOL_TENOR_VEGA` bucket × `σ1(K*, τ) − σ0(K*, τ)`); if the rate curve is an
`InterpolatedRateCurve`, the scalar `rho` row is replaced by
`rho.<tenor>` rows (`RATE_KEYRATE` bucket, per-1bp reported, × Δr(τ) in bp)
plus the parallel reconciliation in metadata. Objects that are not term
structures keep the scalar rows.

## 8. Lifecycle event term

`product_alive_t1` is the contract as it would be priced at t1 had no event
fired between t0 and t1, carrying the **t0** lifecycle state. Rules:

- Lifecycle states equal (or both None) and `product_alive_t1` not given →
  `product_alive_t1 = snapshot_t1.product`.
- Lifecycle states differ and `product_alive_t1` not given →
  `ValidationError` (guessing the alive contract would silently mislabel the
  event row).
- Backtest recorders obtain it from the tracker: `product_for_pricing(t1,
  env_t1)` evaluated before `apply_lifecycle_events` / `process_day`.

The event row is `pv_t1.total − pv_alive_t1.total` with `pv_alive_t1`
valued on (product_alive_t1, engine_t1, env_t1, state_t0, point t1). On a KO
day this is the jump from the contingent claim to the fixed receivable; on a
coupon day it is the difference between the engine's own resolution of the
t = 0 observation and the tracker's determination (near zero when they
agree); on an expiry day the settlement. A position that is already terminal
at t0 has no contingent leg: only `time` and `rate` can be non-zero (the
receivable's carry and discounting), and the event row covers its payment.

## 9. Position and portfolio layers

```python
@dataclass(frozen=True)
class PositionSnapshot:
    position_id: str; underlying: str; snapshot: ValuationSnapshot; entry_price: float

@dataclass(frozen=True)
class BookSnapshot:
    date: datetime
    positions: Mapping[str, PositionSnapshot]
    @classmethod
    def from_portfolio(cls, portfolio, date, *, lifecycle_manager=None) -> "BookSnapshot"

def explain_position(pos_t0: PositionSnapshot, pos_t1: Optional[PositionSnapshot],
                     *, trades=(), product_alive_t1=None, config=None) -> PositionExplainResult
def explain_portfolio(book_t0: BookSnapshot, book_t1: BookSnapshot,
                      *, trades=(), transaction_costs=0.0, config=None) -> PortfolioExplainResult

@dataclass(frozen=True)
class PositionExplainResult:
    position_id: str; underlying: str
    instrument: PnLExplainResult          # market + event rows at q0
    trade_rows: tuple[ExplainRow, ...]
    total_pnl: float                      # q1·pv1 − q0·pv0 − trade cash
    rows: tuple[ExplainRow, ...]          # instrument rows + trade rows

@dataclass(frozen=True)
class PortfolioExplainResult:
    date_t0: datetime; date_t1: datetime
    positions: Mapping[str, PositionExplainResult]
    rows: tuple[ExplainRow, ...]          # aggregated by (method, factor, term) + cost row
    total_pnl: float
    metadata: Mapping[str, Any]           # includes "reconciliation"
    def to_frame(self) -> pd.DataFrame
```

`from_portfolio` deep-copies the environment per underlying and each
position's lifecycle state; products and engines are referenced (the equity
backtest never mutates them day to day except by lifecycle substitution,
which the copied state records).

Rows beyond the per-position market and event rows:

- **trade**: for a surviving position with `q1 ≠ q0` and for a new position,
  `Σ traded_qty × (unit_pv_t1 − trade.price)` from the supplied
  `TradeRecord`s matched by `position_id`. Market rows use `q0`, so
  `q1·pv1 − q0·pv0 − Σ traded_qty·price = market rows + trade rows` holds.
- A position present at t0 and absent at t1 with a terminal lifecycle state
  is explained by its event row. Absent with no terminal state and no closing
  trade record → `ValidationError`.
- **transaction_cost**: the day's costs as one portfolio row.
- **hedge legs** (replay engine): the futures position is delta-one and gets
  `spot` = contracts×multiplier×ΔS, `basis` = contracts×multiplier×(ΔF − ΔS),
  and roll / rebalance trade rows.

`PortfolioExplainResult` holds the per-position results, aggregated rows by
(method, factor, term), `to_frame()` with one row per (position_id or
`"portfolio"`, method, factor, term), and `reconciliation = {"expected":
ΔV − costs + trade cash, "explained": Σ rows, "gap": …, "ok": bool}` in
metadata (asserted with `is_close`, surfaced, never swallowed).

## 10. Backtest integration

Both configs gain `pnl_explain: Optional[PnLExplainConfig] = None`. When
None nothing changes: no extra pricings, `states` / `greeks` frames
byte-identical, goldens untouched.

`PnLExplainRecorder` keeps yesterday's `BookSnapshot` (or per-replay
snapshots) and the day's trade records, and emits rows:

- **Equity `BacktestEngine`** (`_step`): after `_update_pricing_environment`
  and before `_process_lifecycle`, capture alive products (positions'
  current products); after lifecycle and hedging, build today's
  `BookSnapshot`, call `explain_portfolio` with the day's trade records and
  incremental transaction costs, then store today's snapshot as tomorrow's
  t0. Hedge positions are ordinary delta-one positions through the same
  kernel. Executor `realized_pnl` on closed/rolled contracts is matched by
  the closing trade records.
- **`ReplayBacktestEngine`** (`run` loop): per replay, call
  `product_for_date(date, env)` before `apply_lifecycle_events` (alive
  product with t0 state) and again after events with the post-event state;
  keep yesterday's engine reference so the daily recalibrated engine yields a
  `model` row; explain the futures/spot hedge leg (§9). The valuation point
  for numeric-time ledgers is built from the replay start date.

Rows land in `explain_df` on `BacktestResults`, `BookBacktestResults` and
`AutocallableBacktestResults`: columns `date, position_id, method, factor,
term, pnl, greek, cash_greek, step` plus one column per move key, and a
per-day `reconciliation_gap` comparing the row sum with that day's change in
total PnL from the states frame.

## 11. Error handling

| Condition | Behaviour |
|---|---|
| `snapshot.date != pricing_env.valuation_date` | `ValidationError` |
| `date_t1 <= date_t0` | `ValidationError` |
| invalid `PnLExplainConfig` (non-market factor in the order, duplicates, unknown stencil, `clock` with `exact_gap`) | `ValidationError` |
| clock mismatch between environments (day count, bus days, calendar) | `ValidationError` |
| lifecycle state changed, no `product_alive_t1` | `ValidationError` |
| quantities differ in instrument-level `explain` | `ValidationError` |
| position gone at t1, not terminal, no closing trade | `ValidationError` |
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
   tolerance and `extended` shrinks it.
2. **Shapley**: three changed factors; allocation equals the brute-force
   average over all 6 orderings and sums to the sequential total.
3. **Time-step equivalence**: a date-based and a float-maturity build of the
   same contract give the same `time` row.
4. **Route and unit guard**: `BumpConfig(vol_bump=0.02)`, analytical and
   numerical routes agree with dV/dσ×Δσ within FD tolerance; cash columns
   equal `build_cash_greeks_report` values for the base greeks.
5. **Lifecycle days**: snowball KO day and Phoenix coupon day on quick PDE
   and QUAD configs; event row equals `pv_t1 − pv_alive_t1`; a terminal-at-t0
   position explains its receivable carry only.
6. **Backtest reconciliation gate**: both engines on a short window with the
   explain on; per-day row sums equal the day's total-PnL change within
   1e-8 (spot delta hedge, multi-instrument option hedge with a roll, replay
   futures hedge with a roll); with the explain off, `states` / `greeks`
   frames are byte-identical to `main` (same-machine invariant script, like
   the greeks fingerprint gate; not a committed golden).
7. **Clock**: Friday→Monday on a calendar env gives `days=3`, `trading_days=1`;
   `per_step` with `"1td"` uses one step, with `"1d"` three.
8. **Vol-model engine**: LV quick config across a recalibration; `model` row
   carries the surface move and `vol` is near zero.
9. **Bucketed**: term-structure surface and interpolated curve; bucket rows
   sum to the scalar rows within FD tolerance.
10. **Row order**: `to_frame()` column and row order stable under several
    hash seeds (mirrors `test_greeks_key_order.py`).

## 13. Sequencing and gates

| Phase | Content | Gate |
|---|---|---|
| P0 | package skeleton, `ValuationSnapshot`, `value`, `FactorDiff`, waterfall (sequential + shapley), result types | tests 1–3, 10 |
| P1 | Taylor explainer, `resolve_route` helper, unit normalisation, display table, time term modes, bucketed opt-in | tests 4, 7, 9; riskmeasures suite green and its fingerprint byte-identical |
| P2 | lifecycle term, position / portfolio layers, trade / cost / hedge rows | test 5 |
| P3 | recorders in both backtest engines, config fields, `explain_df` | tests 6, 8; backtest and replay suites green |
| P4 | README, root-guide row, demo script | — |

Every phase runs with worktree source shadowing the editable install
(`PYTHONPATH=$PWD`).

## 14. Compatibility contract

- No numeric change anywhere in `riskmeasures`: the only addition is the
  pure `resolve_route` helper.
- Backtests with `pnl_explain=None` are unchanged in behaviour, cost and
  output.
- `PricingEnvironment`, products, engines and lifecycle types are untouched.

## 15. Out of scope

- Other asset classes (the package shape is ready for `fx/` etc.).
- Dynamic-scenario integration (same recorder pattern; separate follow-up).
- Sticky-delta spot convention (rejected upstream in `sticky.py`).
- Model-parameter attribution inside a vol model (per-Heston-parameter rows);
  `model` is one row.
- A report / HTML renderer beyond `to_frame()`.
