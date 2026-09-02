# PnL explain (`quantark.pnlexplain`)

Two-snapshot PnL explain for equity option products: a **full-revaluation
waterfall** (exact, sequential by default, Shapley opt-in) and a
**greeks-based Taylor** explain, both on one factor model, with lifecycle
cash flows as an explicit event term, position / portfolio aggregation with
trade and cost rows, and a daily explain series from both backtest engines.
Design: `docs/superpowers/specs/2026-09-02-pnl-explain-design.md`.

## What it does

A `ValuationSnapshot` pins one position on one date: product, engine, pricing
environment, quantity, optional lifecycle state. `explain(snapshot_t0,
snapshot_t1)` attributes the change of the **value identity**

    value = contingent MTM (quantity × engine.price) + pending receivable PV + paid cash

to the seven market factors `time → spot → vol → rate → dividend → basis →
model`, the lifecycle event, and (for Taylor) an explicit `unexplained`
residual. Both methods read the same `FactorMoves` (scalar moves sampled at
the product coordinate: the strike and the alive-at-t1 tenor) and share one
`ScenarioCache`, so every scenario value of a call is priced once.

## Quick start

```python
from datetime import datetime
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import ExplainMethod, ValuationSnapshot, explain
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

def env(spot, vol, date):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(vol),
                              rate_curve=FlatRateCurve(rate=0.03),
                              div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=date)

engine = BlackScholesEngine()                       # ONE engine object on both sides
fri, mon = datetime(2026, 6, 26), datetime(2026, 6, 29)
s0 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
                       engine, env(100.0, 0.20, fri), date=fri, quantity=10.0)
s1 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0 - 3 / 365),
                       engine, env(103.0, 0.22, mon), date=mon, quantity=10.0)
res = explain(s0, s1)
print(res.to_frame()[["method", "kind", "factor", "term", "pnl", "greek", "cash_greek", "spot_return", "vol_pts", "days"]])
assert abs(res.reconcile(ExplainMethod.WATERFALL)) < 1e-9 and abs(res.reconcile(ExplainMethod.TAYLOR)) < 1e-9
```

The contract at t1 must be the t0 contract **rolled** by the calendar gap
(a float maturity shortened by `days / 365`, or the same date-based
product). Anything else is a contract replacement and raises
`ValidationError`.

## Reading a result

`PnLExplainResult.rows` is a tuple of `ExplainRow`s. Three row kinds:

| kind | meaning | summed? |
|---|---|---|
| `COMPONENT` | an attribution row of one method (`waterfall`, `taylor`) or of both (`shared`) | yes, per method |
| `INFORMATIONAL` | a sub-decomposition (theta sub-rows, key-rate view) | never |
| `SUMMARY` | the `total` row | never |

**Additivity contract:** for a method `M`, the `COMPONENT` rows with method
in `{M, shared}` sum to `total_pnl` (`res.reconcile(M) == 0`). The lifecycle
event row is `shared`; the Taylor `unexplained` row is a `taylor` component
so that method reconciles exactly too. Summing every row is **not** the
total.

Every row carries `pnl` (money, whole position), and Taylor rows also carry
`greek` (the raw position derivative) and `cash_greek` (desk convention, see
the table below). `moves` holds the factor move in display units:
`spot_return`, `vol_pts` (Δσ × 100), `rate_pct`, `div_pct`, `basis_pct`,
`days`, `trading_days` (only with a calendar), `tenor` (bucket rows).

The waterfall `time` row is `time_pure` = V(alive contract at t1, t0 market)
− V(t0) when time is the first step of the order (the default); other orders
sum exactly too but the time row then carries interaction. Metadata reports
`time_pure`, `effective_factors`, the coordinate, `route`, `n_steps`, and
`contract_roll_days`.

## Lifecycle days

For tracker-backed products (snowball, phoenix, barrier families) pass a
`LifecycleTransition(product_alive_t1, engine_alive_t1, state_before,
state_after, events)`: the contract as it would have been priced at t1 had
nothing fired, the engine that prices it, deep copies of the state on both
sides and the events of `(t0, t1]`. The event row is
`V(t1 as booked) − V(alive contract, t1 market)`: KO/KI/coupon/maturity cash,
the contingent leg dropping to zero, and any engine substitution on a KI (a
`model` step needs no event; an engine swap **without** an event is
rejected). Receivable payments are not events: a pending receivable becomes
paid cash inside the `time` step. A terminal position keeps a tombstone
until its cash is paid; its contract is never priced and never rolled.

The lifecycle fingerprint is computed over every state field except the
trackers' bookkeeping (`observed_*_indices`, the `valuation_point` clock
stamp, and the replay's `pending_settlement_cashflow` / `settled` mirrors).
Float-schedule products keep their ledger in contract time; give the
snapshot the tracker's `valuation_point` (the recorders do this for you).

`contract_roll_days=0` on a transition declares, explicitly, that the same
float-maturity contract was repriced without rolling. The equity
`BacktestEngine` does this for every untracked position; the time row then
carries only the valuation-date effect and no contract theta.

## Position and portfolio

`explain_position(pos_t0, pos_t1, trades=...)` handles the §9 case table:
unchanged, quantity change with trades, opened today, closed by trading
(tombstone), terminated by lifecycle (tombstone + transition). Quantities are
**units** (`engine.price` is per unit; multipliers belong to the recorders).
`ExplainTrade(position_id, quantity, price)` uses signed units (buy > 0) and
`cash = −quantity × price`; a trade row is `quantity × (unit PV at t1 −
price)`. Hedge futures legs are `QuotedLegSnapshot`s valued at a quoted
price with contract-specific ids (`hedge:IF2401`), so a roll is a close on
one id and an open on the next. `explain_portfolio(book_t0, book_t1,
trades=, transaction_costs=, transitions=)` aggregates per underlying and
book-wide; display columns survive aggregation only when every constituent
shares one coordinate and step. Books are single-currency (labels are
checked, never converted).

## Backtests

Both engines take `pnl_explain=PnLExplainConfig(...)` on their configs
(`None` leaves the run byte-identical). Results expose `explain_df` (one row
per explain row per day, `date` first) and `explain_reconciliation_df`
(`RECON_COLUMNS`: per day, method and level `expected`, `explained`, `gap`,
`ok`, and at portfolio level `expected_states` / `gap_states`). `expected`
is the value identity; `expected_states` is the engine's own booked PnL
change. `gap` is zero by construction. `gap_states` is zero for the
average-cost executors (multi-instrument equity, replay) and is **reported**
for the equity simple `HedgeExecutor`, which adjusts a hedge's quantity
without re-averaging its entry price (a pre-existing accounting quirk).

## Vol-model engines

The replay engine calibrates a fresh vol-model engine per day; the recorder
passes that day's engine as `engine_alive_t1`, so the recalibration lands in
the `model` row and the surface swap is ~0 (the model prices from its own
calibrated state). MODEL is detected by engine **identity**: two equivalent
engine objects read as a model change, so keep one engine object across
snapshots when nothing changed.

## Conventions (cash columns and move keys)

| Term | Raw | Cash greek | Move keys |
|---|---|---|---|
| delta | dV/dS | Δ·S | `spot_return` |
| gamma | d²V/dS² | Γ·S²/100 | `spot_return` |
| speed | d³V/dS³ | ·S³/10⁴ | `spot_return` |
| vega | dV/dσ | ×0.01 (per 1 vol pt) | `vol_pts` |
| volga | d²V/dσ² | ×10⁻⁴ | `vol_pts` |
| vanna | d²V/dSdσ | ·S·0.01 | `spot_return`, `vol_pts` |
| zomma | d³V/dS²dσ | ·S²/100·0.01 | `spot_return`, `vol_pts` |
| theta family | per step | per day | `days`, `trading_days` |
| charm / color / vega_theta | dΔ/dt, dΓ/dt, dvega/dt | ·S, ·S²/100, ×0.01 per day | + `days` |
| rho / dividend_rho | dV/dr, dV/dq | ×0.01 (per 1%) | `rate_pct`, `div_pct` |
| dividend_volga / delta_q | d²V/dq², dΔ/dq | ×10⁻⁴, ·S·0.01 | `div_pct` (+ `spot_return`) |

PnL always comes from the raw derivatives; the cash columns are display only.
Built-in stencils (`first_order`, `standard`, `extended`) route a vanilla
numerically (vanna / volga and the theta sub-rows have no closed form); an
explicit closed-form stencil such as `["delta", "gamma", "vega", "theta",
"rho"]` routes analytically. `time_term="exact_gap"` (default) uses the exact
`time_pure` as the theta component with `theta_contract` / `ledger_carry` /
`r_theta` / `q_theta` / `convexity_theta` as informational sub-rows;
`time_term="per_step"` with `clock="1d"|"1td"` uses the calculator's
per-step theta × steps.

## Bucketed (opt-in, experimental)

`bucketed=True` replaces the scalar `vega` row by `vega.<τ>` tenor rows on
the t0 pillars (the t1 surface is sampled there) that sum to the scalar
vega. The calculator's key-rate rho is *carry-invariant* (forward held,
dividend re-derived), a different sensitivity from this factor model's rate
step, so `rate_keyrate.<τ>` rows are **informational** beneath the scalar
`rho`, tagged `convention="carry_invariant"`.

## Limitations

- Untracked products carry no cashflow ledger; the equity `BacktestEngine`
  never rolls them (declared via `contract_roll_days=0`).
- Engines are used as given (mutated in place by the backtests).
- TradingClock-wrapped environments are untested.
- The equity simple `HedgeExecutor` entry-price quirk shows up as a non-zero
  `gap_states`; a delayed first-hit barrier settlement cannot be priced by
  `BarrierAnalyticalEngine`.
- Bucketed mode is experimental and never on the default path.
