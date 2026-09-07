# PnL Explain Follow-ups — Patch Spec

**Date:** 2026-09-03
**Status:** draft for review (written autonomously after the parent feature merged; the decisions in §2 are the author's and are flagged where the user may want to overrule)
**Parent spec:** `docs/superpowers/specs/2026-09-02-pnl-explain-design.md` (merged to main at 5f89133)
**Plan:** `docs/superpowers/plans/2026-09-03-pnl-explain-followups.md`

This is a *patch* spec: it fixes the six pre-existing engine and library
quirks that the PnL explain feature exposed and had to work around, and it
amends the parent spec's sentences that describe the work-arounds (§9). It
does not change the parent spec's decisions on methods, factor model,
layers, units or display.

---

## 1. Problem

The final report of the PnL explain feature listed six quirks it found but
did not fix. Each one either forces the module to *report* a gap instead of
closing it, restricts which configurations can be explained, or leaves a
row informational that should be a component.

| # | Finding | Where it lives today | What the explain does today |
|---|---|---|---|
| A1 | The equity simple `HedgeExecutor` changes a hedge's quantity without re-averaging its entry price, so the engine's own `states` PnL jumps by Δq·(p − entry) on every adjust day. | `quantark/backtest/equity/hedge_executor.py` `_update_hedge_position` (calls `update_position(quantity=...)` only) | Reports the difference as `gap_states`; the identity (`gap`) still holds. |
| A2 | With `delta_threshold=0` on a knocked-out barrier book the same executor tries to set its hedge to exactly zero and `Portfolio.update_position` raises "Quantity cannot be zero" — explain on or off. | same, plus `quantark/portfolio/equity/portfolio.py:177` | Cannot run that configuration. |
| B | The equity `BacktestEngine` never rolls an untracked float-maturity product: a `maturity=1.0` vanilla is priced as a one-year option on every day of the run. Only lifecycle trackers roll. | `quantark/asset/equity/lifecycle/manager.py` (`pricing_products` returns `position.product` for untracked ids; `process_day` touches tracked ids only) | Recorder declares `contract_roll_days=0`; the time row carries only the valuation-date effect and `theta_contract` is zero (parent §8). |
| C | `BarrierAnalyticalEngine` (and the one-touch / sharkfin analytical engines) refuse a delayed first-hit payment (`CapabilityError`), even a constant year-fraction lag. | `barrier_analytical_engine.py:92-104`, `one_touch_analytical_engine.py:89-100, 207-222`, `single_sharkfin...:116-126`, `double_sharkfin...:168-178` | The lagged lifecycle backtest fixture must price on a `ConstantEngine` with greeks off. |
| D | The calculator's only key-rate rho is *carry-invariant* (forward held, dividend re-derived: pure discounting). The explain's rate factor replaces the rate curve with the dividend held, so the two sensitivities differ (opposite sign for a call). | `riskmeasures/bucketed_coordinates/rate_keyrate.py` (`CarryInvariantDividendYield` wrap) | `rate_keyrate.<τ>` rows are INFORMATIONAL beneath the scalar `rho`; bucketed mode never replaces the rate component. |
| E | `MODEL` is detected by engine *identity*: two equivalent engine objects read as a model change. | `pnlexplain/equity/factor_diff.py:180` (`engine_alive_t1 is not snap0.engine`) | An extra (zero) MODEL pricing per step, and under Shapley an extra factor dimension. |
| F | TradingClock-wrapped environments are untested. | `param/vol/trading_clock_surface.py`, `param/rrf/trading_clock_curve.py`, `param/div/trading_clock_yield.py`, `util/calendar/trading_clock.py` | Unknown: wrapper equality is object identity, and the wrappers' time maps are anchored at the valuation date, which the waterfall's TIME step moves. |

---

## 2. Decisions

Made autonomously; the "override?" column marks the ones a reviewer is most
likely to want to change. Nothing here re-opens a parent-spec decision.

| Item | Decision | Alternative not taken | Override? |
|---|---|---|---|
| A | Give the simple executor the multi-instrument executor's average-cost accounting (blend on increase, realise on reduce, realise-and-re-enter on flip, close-and-remove on net zero) and a `realized_pnl` attribute. Not opt-in: the old accounting is an error, not a convention; the other two hedge paths already do this. | A flag keeping the old accounting. | Changes `states_df.pnl` on adjust days for every delta-neutral equity backtest. No test pins those numbers. |
| B | Roll untracked **schedule-free** float-maturity products daily inside the shared `PortfolioLifecycleManager`, by the trackers' rule `max(1e-8, m₀ − days/365)` from the day the position is first seen. Whitelist by class: `EuropeanVanillaOption`, `AmericanOption`, `CashOrNothingDigitalOption`. `Futures` are **not** rolled (a static-maturity futures hedge is the documented design of both executors). Schedule-bearing float products (Asian, accumulator, range accrual, DCN, KO-reset snowball, untracked barrier family) keep the status quo and warn once. The equity engine always constructs the manager; `handle_lifecycle_events` gates only the trackers. | Convert float maturities to dates at start; a `roll_float_maturities` config flag. | Changes every equity backtest holding a float-maturity vanilla (all current tests and `example/multi_greek_hedging_demo.py`). Dynamic scenario inherits the roll only when its own lifecycle flag is on. |
| C | Support a **constant** first-hit lag in the analytical first-passage formulas by exact scaling `e^{−r·L}` under the formula's own flat-rate assumption (the parent settlement spec's explicit allowance). Constant means: `YEAR_FRACTION` lags; `CALENDAR_DAYS` lags with `UNADJUSTED` convention under a day count that maps n days to a start-independent fraction (`CALENDAR_DAYS`, `ACT_365` → n/365; `ACT_360` → n/360). Everything else stays a `CapabilityError`. | Approximating business-day lags by an average. | No: the settlement spec forbids averages. |
| D | Add an opt-in `RateKeyrateConvention.DIVIDEND_HELD` to the bucketed-greeks request (default stays `CARRY_INVARIANT`, bitwise unchanged); the explain requests it and promotes the pillar rows to COMPONENT rows that replace the scalar `rho`, exactly as tenor-vega replaces `vega`. | A second coordinate enum member. | Whether the desk wants dividend-held key-rate rho exposed at all outside the explain. |
| E | Engines may declare `MODEL_FINGERPRINT_ATTRS`; `BaseEngine.model_fingerprint()` returns those attributes (sub-engines recursively) or `None`. Equivalence = identity, or same class with equal, normalisable fingerprints. Default `None` keeps identity semantics, so no engine can be judged equivalent by accident; only params-only analytical engines opt in. | Structural comparison of `vars(engine)`. | No: a structural rule would call two local-vol PDE engines with different `_prebuilt` surfaces equal. |
| F | Waterfall support for clock-wrapped environments: wrappers compare by (class, inner, clock) with the map's anchor ignored; every state's wrappers are re-anchored to that state's valuation date; anchors must equal their environment's valuation date; a wrapper on one side only, or two clocks, is a `ValidationError`; float-maturity products on a `BUSINESS_DAYS` environment are rejected. The **Taylor method fails closed** on a wrapped environment because the calculator's vol / dividend / rate bumps replace the wrapper with a calendar-quoted flat object (a different clock). | Wrap-aware bumps in `riskmeasures`. | Yes: the vega unit question (per σ_cal or per σ_td) is a desk decision; recorded in §14. |

---

## 3. A — Simple `HedgeExecutor` average-cost accounting

### 3.1 Behaviour today

`HedgeExecutor._update_hedge_position` computes `new_quantity = old + hedge_size`
and calls `portfolio.update_position(position_id, quantity=new_quantity)`.
The entry price stays at the first fill. `Portfolio.get_portfolio_pnl()`
is Σ (price − entry) × quantity, so on an adjust day the booked PnL moves by
Δq·(p − entry) with no cash to justify it. `close_hedge_position` removes
the position without realising anything, so its PnL simply vanishes from
`portfolio_pnl`. The engine already reads
`getattr(self.hedge_executor, "realized_pnl", 0.0)` into `net_pnl`
(`engine.py:_record_state`), so a `realized_pnl` attribute is picked up
without an engine change. `Portfolio.update_position(quantity=0)` raises,
which is the A2 crash.

### 3.2 Change

Mirror `MultiInstrumentHedgeExecutor._adjust_contract` exactly, with
`price = pricing_env.spot` (the simple executor's fill price) and
`is_zero` from `quantark.util.numerical`:

| Case | Condition | Accounting | `trade_type` |
|---|---|---|---|
| net to zero | `is_zero(new)` | `realized_pnl += (p − entry)·old`; `remove_position`; forget the id | `close` |
| flip | `old·new < 0` | `realized_pnl += (p − entry)·old`; `update_position(quantity=new, entry_price=p)` | `adjust` |
| increase | `abs(new) > abs(old)` | `entry' = (entry·old + p·Δq)/new`; `update_position(quantity=new, entry_price=entry')` | `adjust` |
| reduce | otherwise | `realized_pnl += (p − entry)·(old − new)`; `update_position(quantity=new)` | `adjust` |

`close_hedge_position` realises `(p − entry)·quantity` before removing.
`realized_pnl` starts at 0.0 and is reported in `get_statistics()`. The
trade record's metadata carries `action`, `old_quantity`, `new_quantity`
and `entry_price_after` (`None` after a close).

After a close, `_hedge_position_ids` no longer holds the underlying, so the
next hedge opens a fresh position (fresh id). The explain sees that as a
closing trade on the old id (tombstone) and an opening trade on the new
one, which its §9 rules already handle.

### 3.3 Consequences

- `states_df.pnl` equals the economic identity on every day; the
  recorder's `gap_states` is zero for the simple executor and the parent
  spec's "reported, not gated" carve-out is removed (§9).
- A2 disappears: hedging a book whose delta has gone to zero closes the
  hedge instead of raising.
- Numbers move for every equity backtest that adjusts a spot / futures
  hedge more than once. No test pins them; `test_multi_greek_backtest`,
  `test_hedge_scenarios`, `test_backtest_interface` assert thresholds and
  shapes only.

### 3.4 Tests

- New `test/test_hedge_executor.py`: the four branches with the same numbers
  as `test/test_multi_hedge_executor.py:108-165` (10 @ 100, +10 @ 110 →
  entry 105; −4 @ 110 → realised 20; −26 @ 120 → flip, realised +240,
  entry 120; +10 @ 130 → close, realised −100, position gone, record
  `close`); after every trade `portfolio_pnl + realized_pnl` equals
  Σ qᵢ·(S − pᵢ) over the fills; a hedge after a close opens a new id;
  `close_hedge_position` realises.
- `test/test_pnlexplain_backtest_equity.py`: the "gap documented" test
  becomes "gap is zero"; a new test runs the knocked-out barrier book with
  `delta_threshold=0.0` to completion, finds a `close` trade on the KO day,
  and reconciles with `gap_states` zero.

---

## 4. B — Rolling schedule-free float-maturity products

### 4.1 Behaviour today

`BaseEquityOption.get_maturity` returns the float `maturity` unchanged when
no `exercise_date` is set. The equity `BacktestEngine` prices
`position.product` daily and only `PortfolioLifecycleManager` replaces
products (tracked ids: `tracker.product_for_pricing`, which sets
`maturity = max(1e-8, m₀ − elapsed)`). Untracked float products therefore
never age. `KnockOutResetSnowballOption` already gets a registration
warning saying so. `FuturesHedgeInstrument` documents its float maturity as
"static, consistent with the existing single-instrument futures hedge":
the futures hedge is a constant-maturity proxy by design and stays one.

### 4.2 Change

New module `quantark/asset/equity/lifecycle/float_roll.py`:

```python
FLOAT_ROLLABLE_PRODUCTS = (EuropeanVanillaOption, AmericanOption, CashOrNothingDigitalOption)
MATURITY_FLOOR = 1e-8                          # the trackers' floor

def is_float_rollable(product) -> bool         # class in the whitelist, no exercise_date / maturity_date, float maturity set
def has_unrolled_float_maturity(product) -> bool   # float maturity, no dates, NOT rollable, not a delta-one product

class FloatMaturityRoller:
    register(position_id, product, date)      # first sight wins: (base_date, m0)
    rolled(position_id, product, date)        # deepcopy(product) with maturity = max(FLOOR, m0 - days/365)
    retain(position_ids)                      # forget ids that left the book
```

`PortfolioLifecycleManager` owns one roller:

- `pricing_products(portfolio, date)`: untracked + rollable → `register`
  then the rolled copy (a **new object**, so the recorder's
  `_is_unrolled_float_contract` is false and the parent §5.3 roll check
  validates the roll); untracked + not rollable → `position.product`
  (unchanged behaviour).
- `process_day`: after the tracked branches, untracked + rollable →
  `position.product = rolled`; untracked with a float maturity and no
  roll rule → one `UserWarning` per position id (Futures and the already
  warned KO-reset snowball excluded); then `retain(portfolio.positions)`.
- The roll is measured from the first date the manager sees the position
  (day one for initial positions, so their maturity is "as of the start
  date", exactly the trackers' convention). Positions added later are
  Futures / Spot hedges today and are not rolled.

`BacktestEngine._initialize` always constructs the manager;
`handle_lifecycle_events` gates only `register_positions` (the trackers).
`_process_lifecycle` therefore always runs; with no trackers the ledger is
empty and every lifecycle quantity is 0.0, so runs without lifecycle
products are unchanged except for the roll itself.

### 4.3 Consequences

- A `maturity=1.0` vanilla held for 40 business days ends the run at
  `1 − 55/365`. The explain's `contract_roll_days` metadata equals the
  step's calendar days for these positions and the `theta_contract`
  sub-row is non-zero; the recorder's `contract_roll_days=0` declaration
  remains for Futures hedges and schedule-bearing untracked products (§9).
- A rolled contract reaching the floor stays in the book at intrinsic
  value: a generic expiry transition for untracked products is still out
  of scope (parent §15) and is listed in §14.
- Dynamic scenario shares the manager and rolls when its
  `handle_lifecycle_events` is on (its default); with the flag off it keeps
  constant maturities (§14).
- Numbers move for every equity backtest holding a whitelisted float
  product: `test_multi_greek_backtest`, the pnlexplain equity backtest
  tests (identity-based, unaffected in outcome), `example/multi_greek_hedging_demo.py`.

### 4.4 Tests

- New `test/test_float_maturity_roll.py` (manager level): vanilla rolls by
  days/365 from first sight; product objects are fresh each day but
  `pricing_products` before and `process_day` after agree; floor is
  reached and held; `Futures` and an `AsianOption` are untouched, the
  Asian warns exactly once, the Futures never; ids that left the book are
  forgotten.
- `test/test_backtest_lifecycle.py`: with `handle_lifecycle_events=False`
  the barrier position still survives (unchanged test) **and** a vanilla
  in the same book still rolls (new test).
- `test/test_pnlexplain_backtest_equity.py`: `contract_roll_days ==
  calendar days` and a non-zero `theta_contract` sub-row for the vanilla
  book; `states`/`greeks`/`trades` frames still byte-identical with the
  explain on (existing test).

---

## 5. C — Constant-lag first-hit payment in the analytical engines

### 5.1 Behaviour today

`OneTouchAnalyticalEngine._requests_delayed_hit_payment` returns true for
any non-zero `settlement_convention.lag` or any per-record settlement
timing, and the four analytical engines raise `CapabilityError("...
first-hit payment")` before pricing a not-yet-hit, hit-paid contract.
Already-hit and near-expiry paths resolve the actual date through
`SettlementResolver` and are fine. The parent settlement spec
(`2026-07-29-equity-option-settlement-date-support-design.md`, "Mixed-event
formulas") allows "a constant numeric lag under the formula's flat-rate
assumptions" and forbids average-hit-date or terminal-scaling
approximations.

### 5.2 Change

`quantark/asset/equity/engine/settlement_support.py` gains

```python
def constant_hit_lag_year_fraction(product, pricing_env) -> float
```

returning the lag L ≥ 0 that applies to **every** hit time, or raising
`CapabilityError` (message contains "first-hit"):

| Convention | Result |
|---|---|
| none, or `lag == 0` | 0.0 |
| `YEAR_FRACTION` | `lag` |
| `CALENDAR_DAYS`, `UNADJUSTED`, env day count `CALENDAR_DAYS` or `ACT_365` | `lag / 365` |
| `CALENDAR_DAYS`, `UNADJUSTED`, env day count `ACT_360` | `lag / 360` |
| `CALENDAR_DAYS` with an adjusting business-day convention | `CapabilityError` (holiday-dependent) |
| `CALENDAR_DAYS` under any other day count (`ACT_ACT_*`, 30/360, `ACT_365L`, `BUSINESS_DAYS`) | `CapabilityError` (start-date dependent) |
| `BUSINESS_DAYS` lag unit | `CapabilityError` |
| per-record `settlement_date`, or `settlement_time ≠ observation_time` on any record | `CapabilityError` (unchanged rule) |

The `CALENDAR_DAYS`/`ACT_365` rows are exact because
`calculate_year_fraction` computes `days/365` for both, which is also what
the lifecycle ledger uses for the realised cashflow after the hit.

Pricing: a hit-paid rebate `R` with first-passage time τ under the
formula's flat rate r has value `E[R·e^{−r(τ+L)}·1{τ≤T}] = e^{−rL}·E[R·e^{−rτ}·1{τ≤T}]`.
So:

- `OneTouchAnalyticalEngine._one_touch_price(..., hit_lag)` returns
  `rebate · exp(−rate·hit_lag) · instant_touch_term` for pay-at-hit;
  `price()` computes `hit_lag` with the helper only on the not-yet-hit,
  monitored, pay-at-hit path (replacing the guard);
  `_requests_delayed_hit_payment` is removed.
- `BarrierAnalyticalEngine` and `SingleSharkfinOptionAnalyticalEngine`
  call the helper where their guard was (early, fail-closed) and price the
  rebate leg through the one-touch engine as today, which now applies the
  factor. The BGK-shifted discrete path is the continuous formula on a
  shifted barrier, so the same factor applies.
- `DoubleSharkfinOptionAnalyticalEngine`: the continuous hit-paid cash leg
  (`_continuous_hit_discount_factor`) is multiplied by `exp(−rate·hit_lag)`
  and the guard becomes the helper. Its discrete hit-paid path already
  discounts each observation node at the node's *resolved* settlement time
  through the curve (`_discrete_hit_discount_factor` uses
  `pricing_env.get_discount_factor(record.settlement_time)`), which is
  exact for any lag the schedule resolver can date; it is left unchanged
  and a test pins that a lagged discrete double sharkfin prices as the
  per-node delay says.
- `rate` is the formula's `pricing_env.get_rate(maturity)`; under a
  term-structure curve the factor inherits the flat-r assumption the
  hit-paid leg already makes. Documented in the docstring.

### 5.3 Tests

- `test/test_barrier_family_settlement.py`: the two "rejects
  unrepresentable lag" tests become "constant lag scales by
  `exp(−r(T)·L)`" (rel 1e-12) for one-touch, barrier and single sharkfin;
  a `BUSINESS_DAYS` lag and an `ACT_ACT_ISDA` calendar-day lag still raise
  with "first-hit"; the double-sharkfin continuous leg scales likewise.
- PDE cross-check: `BarrierPDESolver` vs the analytical engine on a
  continuous up-out call with a hit-paid rebate under a flat rate; the
  *lag effect* (unlagged − lagged) agrees between the engines within 5 %
  (the grid error cancels in the difference; the measured values are
  recorded in the test).
- `test/test_backtest_lifecycle.py`: a sibling of the delayed-settlement
  test prices on `BarrierAnalyticalEngine` with greeks on and books the
  same pending PV `20·e^{−0.05·2/365}` on the KO day.
- `test/test_pnlexplain_backtest_equity.py`: the `lagged=True` fixture uses
  the analytical engine with greeks on and still reconciles every day.

---

## 6. D — Dividend-held key-rate rho

### 6.1 Behaviour today

`rate_keyrate.calculate_points` bumps one zero-rate pillar (central, 1 bp)
and wraps the dividend yield in `CarryInvariantDividendYield`, which adds
the same bump to q so F(0,T) is unchanged: pure discounting (desk
convention, bucketed-greeks spec WP3.3). The explain's RATE factor
replaces the rate curve with the dividend held (parent §5.3), i.e. the
forward moves. For a vanilla call the two derivatives have opposite signs,
so parent §7.6 keeps `rho` as the component and lists the buckets as
informational.

### 6.2 Change

`riskmeasures`:

```python
class RateKeyrateConvention(Enum):
    CARRY_INVARIANT = "carry_invariant"    # forward held, q re-derived pointwise (default, unchanged)
    DIVIDEND_HELD = "dividend_held"        # dividend yield held, the rate curve alone moves

BucketedGreeksRequest.rate_keyrate_convention: RateKeyrateConvention = CARRY_INVARIANT   # appended after allow_partial
```

`rate_keyrate.calculate_points` branches in `_rate_bumped_env`: under
`DIVIDEND_HELD` the bumped environment is `deepcopy(env)` with only
`rate_curve` replaced. Every point's metadata gains `"convention"` and the
`rebuild_rule` text names the convention; the calculator's result metadata
gains `rate_keyrate_convention`. The default path is bitwise unchanged.

`pnlexplain/equity/bucketed.py` requests `DIVIDEND_HELD`. The
`rate_keyrate.<τ>` rows become `COMPONENT` rows with
`metadata["convention"] = "dividend_held"`, `covered` includes
`Factor.RATE`, and `taylor.py` puts them in the scalar `rho` slot exactly
as it does for tenor vega; the scalar `rho` row is not emitted.
`rate_keyrate.parallel` stays `INFORMATIONAL` with `sum_of_buckets` /
`reconciles`. `metadata["bucketed_factors"]` becomes `("rate", "vol")` when
both term structures are present.

### 6.3 Tests

- `test/test_rate_keyrate_buckets.py`: under `DIVIDEND_HELD` the bumped
  environment keeps `div_yield` the same object and only the pillar rate
  moves; on a flat curve the dividend-held parallel derivative equals the
  Black–Scholes analytical rho (K·T·e^{−rT}·N(d₂)) within 1e-6 relative and
  is positive for a call while the carry-invariant one is negative; the
  default request's points are unchanged (existing tests untouched).
- `test/test_pnlexplain_bucketed.py`: `rho` absent, pillar rows COMPONENT
  and tagged `dividend_held`, `bucketed_factors == ("rate", "vol")`,
  parallel row informational; with a **parallel** t1 rate move the bucket
  sum matches the scalar `rho` PnL within 5 %; Taylor still reconciles
  exactly; the scalar `delta`/`gamma`/`theta` rows are unchanged by the
  opt-in.

---

## 7. E — Semantic MODEL identity

### 7.1 Change

`BaseEngine`:

```python
MODEL_FINGERPRINT_ATTRS: Optional[Tuple[str, ...]] = None   # class attribute; None = identity only

def model_fingerprint(self) -> Optional[tuple]:
    """(module, qualname, ((attr, value), ...)) or None. Sub-engine values are replaced by
    their own fingerprint; any None makes the whole result None."""
```

Opt-ins are the analytical engines whose instance state is exactly their
construction arguments (verified per engine during implementation by
listing `vars(engine)`): `BlackScholesEngine ("params",)`,
`DeltaOneEngine ("params", "use_market_price")`,
`DigitalOptionAnalyticalEngine`, `BarrierAnalyticalEngine`
(`("params", "_bs_engine", "_one_touch_engine")`),
`OneTouchAnalyticalEngine ("params", "_digital_engine")`,
`DoubleBarrierOptionAnalyticalEngine ("params", "_bs_engine")`,
`SingleSharkfinOptionAnalyticalEngine ("params", "_barrier_engine",
"_one_touch_engine")`, `DoubleSharkfinOptionAnalyticalEngine ("params",
"max_terms", "quad_points", "_double_barrier_engine")` and
`AmericanOptionAnalyticalEngine` with its `method` attribute. MC, PDE, QUAD
and vol-model engines keep `None`. The verification step lists
`vars(engine)` for each candidate and refuses the opt-in when any
instance attribute is missing from the declaration.

`pnlexplain/equity/fingerprints.py`:

```python
def engines_equivalent(a, b) -> bool:
    # a is b, or same class with model_fingerprint() not None on both and _normalize-equal;
    # a fingerprint the normaliser cannot represent counts as NOT equivalent
```

Used by `build_factor_moves` (MODEL changed iff not equivalent), by
`resolve_transition`'s "engine substitution without an event" rule, and
recorded as `metadata["model_equivalent"]`. `ScenarioCache` still keys
bump contexts by object.

### 7.2 Tests

- Two fresh `BlackScholesEngine()` → MODEL not changed, no MODEL state
  priced; `EngineParams(bus_days_in_year=244)` vs default → changed; two
  fresh MC engines → changed (identity kept); a transition whose
  `engine_alive_t1` is an equivalent object is accepted.

---

## 8. F — TradingClock-wrapped environments

### 8.1 Behaviour today

`TradingClockVolSurface(inner, time_map)`, `TradingClockRateCurve`,
`TradingClockDividendYield` re-express a market object on the other time
axis through a `BusinessTimeMap` anchored at the environment's valuation
date ("the map's anchor_date must equal the environment's
valuation_date"). The explain compares market objects with `==`
(dataclass equality for the surface, identity for the two curve
wrappers), so wrappers rebuilt daily always read as changed, and the
waterfall's TIME step moves `valuation_date` while the state still holds
t0-anchored maps. The Taylor method's vol / dividend / rate bumps replace
the surface or curve with a flat calendar-quoted object
(`bump_envs.py:118,143`, `numerical.py:282`), which under a wrapper is a
different clock.

### 8.2 Change

Library (additive): `BusinessTimeMap` stores `horizon_date` and gains
`re_anchored(anchor_date)` (same clock, horizon, `extend_weekdays`); the
three wrappers gain `with_time_map(time_map)`.

`pnlexplain/equity/clock.py` (new):

- `is_clock_wrapped(obj)`: has `inner` and `time_map`.
- `wrapped_equal(a, b)`: same class, `clocks_equal(a.time_map.clock,
  b.time_map.clock)` (`days_per_year` equal, calendars semantically equal
  via `calendars_equal`) and `objects_equal(a.inner, b.inner)`. The
  anchor is **not** compared: moving it is the TIME factor's business.
- `validate_clock_env(env)`: every wrapped field's `time_map.anchor_date`
  equals `env.valuation_date`, else `ValidationError`.
- `validate_clock_pair(e0, e1)`: the same fields are wrapped on both
  sides and their clocks are equal, else `ValidationError` ("a clock
  change is not a market move").
- `re_anchor(env)`: for each wrapped field whose anchor differs from
  `env.valuation_date`, replace it by `with_time_map(time_map.re_anchored(env.valuation_date))`.

Semantics: TIME = the same inner objects and the same clock seen from the
new date (sticky trading tenor), which is what the calendar-axis term
structure already does under TIME; a state that applies VOL without TIME
(Shapley) re-anchors e1's wrapper back to t0 by the same rule. So
`ScenarioCache.build_state` calls `re_anchor(env)` after the factor
swaps, for every state. Change detection uses `wrapped_equal` when both
objects are wrapped.

Fail-closed rules added to `validate_pair`: the two clock validations
above, and "a float-maturity product on a `BUSINESS_DAYS` environment"
(its maturity would be trading years, which the days/365 roll rule does
not describe). `taylor_rows` raises `ValidationError` when any of the
three fields is wrapped, naming the bump-replacement reason; users on a
clock environment pass `methods=(ExplainMethod.WATERFALL,)`.

### 8.3 Tests (`test/test_pnlexplain_trading_clock.py`)

Fixtures from `test/test_trading_clock_axis_equivalence.py` (CHINA_SSE,
D = 244) with a date-based European vanilla:

- calendar axis, holiday-free and CNY-straddling steps: the waterfall
  reconciles; an unchanged inner surface (fresh wrapper, fresh map) is not
  a VOL change and is not priced; the TIME row equals the direct
  revaluation with a re-anchored map; Shapley reconciles.
- axis equivalence: the same market on the trading axis (wrapped curves,
  native vol) gives the same TIME / SPOT / VOL / RATE row PnLs to 1e-9
  (analytical), including a step where the inner σ_td moves.
- anchor ≠ valuation date, wrapper on one side only, two clocks, and a
  float-maturity product on a `BUSINESS_DAYS` environment each raise
  `ValidationError`; the Taylor method on a wrapped environment raises.

---

## 9. Amendments to the parent spec

Applied in place with an "(amended 2026-09-03, patch spec §…)" marker:

| Parent section | Today | After |
|---|---|---|
| §5.3 change detection | "`MODEL` is changed when `engine_alive_t1 is not engine_t0`." | "… when the engines are not equivalent: identity, or the same class with equal `model_fingerprint()`s (§7 of the patch spec); engines without a fingerprint compare by identity." Market objects that are clock wrappers compare by (class, inner, clock). |
| §5.3 factor ownership / §11 | clock fields must compare equal | plus: wrapped fields' anchors equal their valuation date; the same fields wrapped on both sides; float maturities rejected on `BUSINESS_DAYS` environments; Taylor rejected on wrapped environments. |
| §7.6 rate rows | "Rate rows are informational … A q-held key-rate rho would need new numerics in `riskmeasures`, which this feature does not touch." | Rate rows are COMPONENT rows under the dividend-held convention and replace the scalar `rho`; `rate_keyrate.parallel` stays informational. |
| §8 `contract_roll_days` | "the equity `BacktestEngine` does this for every untracked position (it never rolls a float maturity; only lifecycle trackers roll)" | "… for untracked positions without a roll rule (Futures hedges by design, schedule-bearing float products); schedule-free float contracts are rolled by the manager and validated by the ordinary roll check." |
| §10 reconciliation | "The two expectations differ only where … the equity engine's simple `HedgeExecutor` adjusts a hedge quantity without re-averaging the entry price … the `gap_states` gate is asserted for the multi-instrument executor and the replay engine." | "`expected_states` equals `expected` for all three hedge paths; `gap_states` is gated everywhere." |
| §15 out of scope | — | remove "TradingClock untested" from the README limitations; add "Taylor on clock-wrapped environments" and "dynamic-scenario roll with lifecycle handling off" (§14 here). |

`quantark/pnlexplain/README.md` (Backtests, Bucketed, Vol-model engines,
Limitations) and `quantark/backtest/CLAUDE.md` (untracked; simple executor
now average-cost, float roll) are updated in the same task.

---

## 10. Compatibility contract

- `riskmeasures`: the default `BucketedGreeksRequest` and every scalar
  greek are bitwise unchanged; the new convention is opt-in. The engine
  fingerprint hook is additive.
- Parameter objects: `BusinessTimeMap.horizon_date`, `re_anchored`,
  `with_time_map` are additive; no existing method changes.
- Analytical engines: prices of contracts without a first-hit lag are
  bitwise unchanged (the factor is `exp(0) = 1.0`, applied as a multiply
  only on the lagged branch so the unlagged code path is untouched).
- Backtests: `pnl_explain=None` still changes nothing *relative to the new
  engine behaviour*; the explain-on / explain-off byte-identity test keeps
  passing. The engine numbers themselves move for (A) adjust days and (B)
  float-maturity positions, both documented above.
- `PortfolioLifecycleManager.process_day` / `pricing_products` signatures
  are unchanged; the manager gains the roller and the warning.

---

## 11. Error handling

| Condition | Behaviour |
|---|---|
| first-hit lag not constant (business-day lag, adjusted calendar-day lag, start-dependent day count, per-record timing) | `CapabilityError` containing "first-hit" |
| `rate_keyrate_convention` not a `RateKeyrateConvention` | `ValidationError` |
| clock wrapper anchored away from its environment's valuation date | `ValidationError` |
| wrapper on one side only, or two clocks (`days_per_year` / calendar) | `ValidationError` |
| float-maturity product on a `BUSINESS_DAYS` environment | `ValidationError` |
| Taylor method with a wrapped vol / rate / dividend object | `ValidationError` |
| engine fingerprint not normalisable | treated as not equivalent (MODEL changed), never raises |
| hedge quantity nets to zero | position closed with a `close` trade (was: `ValidationError`) |

No new fallbacks: where a lag, a roll or a clock cannot be represented
exactly, the code raises.

---

## 12. Testing gates

| Item | Gate |
|---|---|
| A | `test_hedge_executor.py` green; `gap_states` zero for the simple executor; `delta_threshold=0` KO book runs |
| B | `test_float_maturity_roll.py` green; vanilla maturity decays in the engine; `contract_roll_days == days` in the explain; explain-off byte identity holds |
| C | constant-lag scaling to 1e-12 on four engines; PDE lag-effect cross-check within 5 %; lagged backtest fixture on the analytical engine reconciles |
| D | default bucketed points bitwise unchanged; dividend-held parallel = analytical rho to 1e-6; explain rate buckets replace `rho` and reconcile |
| E | equivalence tests; explain and lifecycle suites green |
| F | `test_pnlexplain_trading_clock.py` green including axis equivalence to 1e-9 |
| all | full suite green on the final tree; `example/pnl_explain_demo.py` still reconciles exactly |

---

## 13. Sequencing

Six independent tasks plus a documentation task, in the order A, B, C, D,
E, F, docs. A and B touch the equity backtest and are done first so the
later backtest tests run on the corrected engine; C is independent; D and
E are library-plus-explain pairs; F is the largest and last. Each task is
reviewable on its own.

---

## 14. Out of scope and findings recorded

Not done here, listed so they are not lost:

- **Taylor on clock-wrapped environments.** Needs wrap-aware bumps in
  `riskmeasures` and a unit decision: the factor model's VOL coordinate is
  σ at (K, T1) read through `env.get_vol`, i.e. σ_cal on the calendar
  axis, while a bump of the inner surface is per σ_td; the two differ by
  √(τ_td/τ_cal). (The rho half of this is resolved: since 2026-09-07
  `build_rate_bumped_env` shifts the calendar-quoted inner of a
  `TradingClockRateCurve` and re-wraps it, so rho is per calendar year on
  both axes.) Fails closed for now pending the vega unit decision.
- **Scalar numerical rho replaced the whole rate curve** with
  `FlatRateCurve(r(T) + bump)` (`numerical.py:282`), so for a
  term-structure curve it measured a curve *reshaping*, not a parallel
  shift: the bumped curve differs from the base by r(T) − r(t) + bump at
  every t < T. Measured on DCN_A with a 1%/3%/5% linear curve the library
  rho was 98× the parallel-shift rho and flipped sign on the inverted
  curve; on a flat curve the two are bitwise equal, which is why no test
  caught it. **FIXED 2026-09-07 as a separate commit on this branch:**
  `bump_envs.build_rate_bumped_env` mirrors the vol/div helpers — a flat
  curve keeps `FlatRateCurve(rate + shift)` (legacy floats, bitwise), any
  other curve is wrapped in `ParallelShiftRateCurve` (the primitive the
  key-rate parallel reconciliation point already uses), a
  `TradingClockRateCurve` gets its calendar-quoted inner shifted. Routed
  through it: `numerical_rho`, `EquityPosition.get_trade_greeks` (central)
  and `get_trade_risk`, and the execution `rate_up` bump cell — the three
  parity-pinned siblings carried the same code. Tests:
  `test/test_rho_term_structure_bump.py`. Bond/rate-engine DV01 bumps use
  the same flat replacement and are out of scope here.
- **Simple executor futures hedges are filled at spot**
  (`hedge_price = pricing_env.spot` for `hedge_instrument_type="futures"`)
  while the position is marked at the forward, so the open day books
  (F − S)·q. The explain's identity still holds (trade at S, MTM at F).
- **Generic expiry / exercise for untracked products** (parent §15): a
  rolled vanilla at the floor stays in the book at intrinsic value.
- **Dynamic scenario with `handle_lifecycle_events=False`** keeps
  constant float maturities; making its manager unconditional is a
  three-line change left for that module.
- **Per-record schedule settlement lags** in the one-touch / barrier /
  single-sharkfin analytical engines stay unsupported (the BGK-shifted
  continuous formula has no per-node hook); the double sharkfin's discrete
  path already resolves them per node.
