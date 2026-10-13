# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

## [Unreleased]

FX TARN / TARF mid-life booking, dated form and ageing (spec:
`docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md`). A target-redemption contract carries its realized
state, can be booked by dates, ages through `time_shift` / `record_passed_fixings`, and is aged day by day in the FX
backtest and FX dynamic scenario.

### Added
- `quantark.asset.fx.product.option.target_redemption_state`: `TargetRedemptionState` (amount banked towards the
  target, determined `PendingPayment`s, dated `as_of`), `TargetRedemptionEvent` / `TargetRedemptionEventType`,
  `TerminalBookingWarning`, `ResolvedTargetRedemption`, and `TARGET_HIT_RTOL` (1e-12, moved from the two TARN/TARF
  engines with its value unchanged).
- Dated `FxTargetRedemptionNote` / `FxTargetRedemptionForward`: `fixing_dates` (the whole contractual schedule),
  `pay_dates`, `state.as_of`; the TARN derives its periods from `initial_date` (or explicit `accrual_factors`) in its
  `day_count_convention` (default ACT/365). Times resolve in the environment's convention; a numeric contract and its
  dated twin price identically. A fixing or pay date that resolves to zero time in the environment's day count (a
  weekend date under `BUSINESS_DAYS`, valued on the Friday) is live, as for its numeric twin (user decision OQ-2 b,
  2026-10-10): the fixing is decided at the quoted spot and its cash is due that day. A dated state counts its banked
  fixings (`TargetRedemptionState.fixings_banked`, fixing identity in schedule order; a booking stating only `as_of`
  banks the fixings dated on or before it); `as_of` never passes the valuation date (user ruling OQ-3, 2026-10-10).
- `resolve(fx_env)` (the engines' one input), `lifecycle_events`, `is_redeemed`, `is_terminal`, `hit_tol`,
  `pending_pv(fx_env)` on both products.
- TARN / TARF take part in the missing-fixing fill protocol (`missing_fixing_fills`, `with_fixings_recorded`,
  `with_missing_fixings_filled`, user decision 2026-10-09): an unbanked passed fixing's RATE is recorded once at the
  base environment's quoted spot, on a copy, so `BaseFxEngine.calculate_greeks`, VaR, stress, dynamic scenarios and
  backtests bank it at the base spot under every bump and shock (one `MissingFixingWarning` per call, none per bump).
  Banking (`record_passed_fixings`, `resolve`) uses a recorded rate in place of the quoted spot; a rate recorded for a
  fixing still after valuation is refused with `ValidationError`.
- `FxTargetRedemptionNote(accrual_end_dates=...)` (dated form; user ruling 2026-10-10 on R26, option (b)): each
  period's own accrual end date, separate from its fixing date. Period `i` accrues from `initial_date` (first) or the
  previous period's end to its own end in the contract day count, so periods may share a fixing date without explicit
  `accrual_factors`. One end per fixing, strictly increasing, each between its fixing and pay dates; it requires
  `initial_date` and excludes `accrual_factors`. Without it the ends are the fixing dates: every existing booking keeps
  its factors and value exactly; such a contract still cannot share a fixing date (a zero-length period).
- FX TARN / TARF take part in run fixings (`quantark.util.run_fixings`, W6): `has_run_fixings`,
  `open_fixings(fx_env)`, `with_time_elapsed(dt)` (a numeric copy aged by `time_shift`, times snapped to exactly 0.0,
  nothing banked), `with_history_settled(fx_env)` (recorded history banked and cash due paid, on a copy) and
  `matured_before(fx_env)`. In the FX backtest and FX dynamic scenario they age with the run's clock: a fixing is
  recorded at the spot of the first instant it was live (a fixing between path instants at the previous instant's,
  `gap=True`), banked in order with its cash; a contract whose last fixing is before the run start is removed with
  `MaturedBeforeRunWarning`. A dated booking is validated against the run's environment first (`_check_as_of`), so a
  stale invalid one (a state claiming an `as_of` or banked fixing after the run start, or pending cash already past)
  raises `ValidationError` at registration instead of being dropped with the warning; numeric bookings are unchanged.
  Moved values: none (a refusal replaces a warning on invalid bookings only).
- `RunFixingRecorder` calls an optional product hook `with_history_settled(env)` after the first valuation's fill and
  after each clock move's recording (`settle_history`); products without it are unchanged.
- `target_redemption_state.booked_cash(product_events)`: the quantity-scaled PAYMENT cash of one run step's
  `RunFixingDay.product_events`.

### Changed
- **Breaking:** `FxTargetRedemptionNote` and `FxTargetRedemptionForward` take a mandatory keyword `state`
  (`TargetRedemptionState(0.0)` for a contract booked at inception); omitting it raises `ValidationError`.
- **Breaking:** a numeric `FxTargetRedemptionNote` states when its first listed period began: `accrual_start` (years
  from valuation) or explicit `accrual_factors`, exactly one. The old default (`t_0 = 0`, the first period starting at
  the valuation instant) halved the first coupon of a contract rebuilt mid-life. `accrual_start=0.0` reproduces the old
  factors bitwise.
- A non-positive fixing time is accepted (an unbanked past fixing) and an empty `fixing_times` is a terminal booking
  (`TerminalBookingWarning`, valued at the PV of its pending cash).
- FX finite-difference theta (`BaseFxEngine._fdm_theta`) steps a numeric (`maturity=`) product by the environment's
  year fraction of `theta_days` calendar days instead of `theta_days / 365` (user decision Q7). Unchanged to the bit
  under `CALENDAR_DAYS` / `ACT_365`; under `BUSINESS_DAYS` a Friday or holiday theta is 0.0 and `ACT_360` steps 1/360,
  matching the dated product. Values that move are listed in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`.
  **Breaking:** `calculate_greeks` on any numeric FX product under a `BUSINESS_DAYS` environment with no calendar now
  raises `ValidationError` in theta (it used to step `theta_days / 365`); a business-day count needs the calendar.
- FX local-vol and Heston theta (`fx_local_vol_model_greeks`, `FxHestonPDESolver`, `FxHestonSLVPDESolver`, and the LV/Heston
  analytical and MC engines that share the first) step a numeric (`maturity=`) product by the environment's year
  fraction of `theta_days` calendar days, as `BaseFxEngine._fdm_theta` does, instead of `theta_days / 365`. The
  `min(step, 0.5 * maturity)` cap is kept and theta is rescaled to a per-calendar-day figure; a zero step (BUSINESS_DAYS
  Friday or holiday) gives theta 0.0 and BUSINESS_DAYS without a calendar raises `ValidationError`. Moved values:
  none under `CALENDAR_DAYS` / `ACT_365` (bitwise); on a 1-year ATM EUR/USD call (notional 1e6, flat 10% vol) `ACT_360`
  local-vol theta -97.0526 -> -98.4012 and BUSINESS_DAYS-on-a-Friday -97.0526 -> 0.0 (coarse-grid Heston PDE 434.64 ->
  432.38 and 0.0; Heston-SLV PDE 462.84 -> 460.54 and 0.0).
- FX TARN / TARF cash due on the valuation date stays in the value (user decision OQ-1 b, 2026-10-10, as W8 values a
  dated contract on its date at its settled value): determined cash is booked (a `PAYMENT` event, out of the value)
  once its pay time is strictly past. A numeric pending `pay_time` of 0.0 and a dated pending `pay_date` equal to
  `as_of` are accepted. Moves: the price on a pay date gains the cash due that day, theta the day before a pay date
  gains it and theta on the pay date loses it (values in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`); inside runs, cash is booked
  once its pay date is past (on the next path instant). "Due" is decided by resolved TIME in the environment's day
  count, also when a date carries a time of day: under a whole-day count, cash due at 00:00 is still due at 15:00
  that day, and 10:00 cash at the next midnight. `TargetRedemptionState` no longer compares a dated `pay_date` with
  `as_of` (it cannot know the day count); the product refuses pending cash strictly past at `as_of` by resolved time
  when an environment prices, records or settles it (`ValidationError`, the same refusal moved from construction).
- FX TARN / TARF periods may share a fixing date or time (user decision OQ-4 b, 2026-10-10): `fixing_dates` /
  `fixing_times` must be non-decreasing instead of strictly increasing. Both MC engines simulate each distinct fixing
  time once and every period sharing it fixes at that spot; periods are accumulated in schedule order (a period after
  the target hit is dead), each with its own pay date and cash. A dated TARN on the default period derivation
  (`initial_date`, then each previous fixing date) cannot share a fixing date (a zero-length period): it states its own
  accrual. The default derivation refuses any non-positive derived factor (two fixings on one day under ACT/365, at
  construction; a zero-business-day period under a `BUSINESS_DAYS` contract count, at resolve), which a dated contract
  priced before. A dated state's banked count (`fixings_banked`, or derived from `as_of`) may not stop inside the
  periods fixing on one calendar date unless the target was hit. `mean_redemption_fraction` does not count a hit at the final fixing time as early.
- FX TARN / TARF: a fixing at exactly the valuation instant (t = 0) is live (run-fixings D1):
  `missing_fixing_fills` / `with_missing_fixings_filled` list only strictly past fixings, so the Greek freeze leaves
  today's fixing to move with the bumped spot, and pricing decides it at the quoted spot without a
  `MissingFixingWarning`. Numeric protocol indices and event labels count from the booking (fixings banked since
  construction + listed position).
- Run fixings (`quantark.util.run_fixings.step_fixings`): a record that was live is recorded first, at its first live
  instant's spot, in index order; only then are the remaining (gap) records probed at the previous instant's spot. A
  TARN / TARF on a business-day clock whose weekend rows quote a spot other than Friday's now records its Friday and
  Saturday fixings at Friday's spot (was a `KeyError`, a gap at the weekend spot, or a split shared-date refusal).
  W6's range accruals and Asians record exactly as before. `ProductReplay` refuses a TARN / TARF (`ValidationError`):
  it does not settle product-held history.
- `FXBacktestEngine` books TARN / TARF cash once its pay date is past (on the next path instant): rows gain
  `paid_cash` (cumulative, quantity-scaled) and `pending_pv`; `portfolio_value` and `final_value` include paid cash.
  A book without TARN / TARF is unchanged (cash columns 0.0).
- `FXDynamicScenarioEngine` books TARN / TARF cash on a dated `DayPath` (determined cash moves to `paid_cash` /
  `realized_cash` once its pay date is past (on the next path instant); one `FIXING` lifecycle snapshot per banked
  fixing; `pending_receivable_pv` filled). An undated `DayPath` with a TARN / TARF in the book raises
  `ValidationError` (user decision Q8 a). Books without TARN / TARF are unchanged.

### Fixed
- FX TARN / TARF theta within one day of a fixing raised (`Time to maturity must be non-negative` / `times must be
  strictly increasing and positive`); the stepped-over fixing is now banked at the held spot and theta is finite.
- Pricing a TARN / TARF after a bare `time_shift` past a fixing raised; the passed fixing is now filled at the quoted
  spot on a copy with one `MissingFixingWarning`.
- A rebuilt mid-life contract no longer prices with the full target: the engines start from `state.accumulated` and
  add the PV of `state.pending`. Inception prices and Greeks are unchanged to the bit.

Accrual time, separate from observation time (spec: `docs/superpowers/specs/2026-10-08-accrual-time-design.md`).
An autocallable's observation is a trading-day quantity (when the market is checked); its accrual is a calendar
quantity in the contract's interest basis (how much it owes). Every snowball, Phoenix and KO-reset record now
carries its own immutable `AccrualPeriod`, derived once at construction, and every product, engine and intraday
path reads accrual through one resolver (`quantark.asset.equity.product.option.accrual`). Ageing (`time_shift`),
snapping an observation time or rebasing a schedule moves observation timing and never moves cash. A numeric
record's accrual is its quoted fraction (`k/12` accrues exactly 1/12 per period).

### Added
- Run-fixings protocol on `RangeAccrualOption`, `FxRangeAccrualOption` (and its foreign / quanto subclasses) and
  record-schedule `AsianOption`: `has_run_fixings`,
  `open_fixings(env)` and `with_time_elapsed(dt)` (the run clock's step: a numeric contract's record times and
  maturity move by `dt`, snapped to exactly 0.0 when zero; records nothing; a dated contract returns itself).
  `quantark.util.fixings.snap_time`.
- `quantark.util.run_fixings` (spec `docs/superpowers/specs/2026-10-09-run-fixings-design.md`): `RunFixingRecorder`
  (`register` at a run's first valuation = W5b's freeze; `advance` at every later path instant records each fixing
  the clock moved past at the spot of the first instant it was live, or at the previous instant's spot with
  `gap=True` when no instant fell on it; numeric contracts age by the environment's year fraction; time-of-day
  instants and a backwards clock are refused), the stateless `step_fixings`, `update_live`, `participates`,
  `run_step_dt`, `at_spot`, `require_date_instant`, `RecordedFixing`, `RunFixingDay`.
  Optional product hooks `matured_before(env)`, `lifecycle_events` and `pending_pv(env)` feed `RunFixingDay.product_events` /
  `pending_pv` (for FX TARN / TARF, W4b).
- `AccrualPeriod`, `AccrualBasis`, `KnockInAnchor` and the resolver: `ko_accrual_factor`, `coupon_period_fraction`,
  `contract_tenor`, `knock_in_anchor_for`. `ObservationRecord.accrual` holds a record's period.
- `AccrualConfig.accrual_end` (`AccrualEndType.OBSERVATION`, the default, or `SETTLEMENT`: a record accrues to its
  authoritative payment endpoint, resolved from the contract's own terms).
- `AccrualConfig.post_accrual_factors`: KO-reset schedule 2's explicit factors (`accrual_factors` is schedule 1's).
- `AccrualConfig.accrual_calendar` / `accrual_bus_days_in_year`: the frozen terms of a `BUSINESS_DAYS` accrual; no
  accrual reads the pricing environment.
- `AutocallableLifecycleState.knock_in_anchor`, set at the knock-in from the record whose test fired, carried on
  pricing copies as `_otc_lifecycle_knock_in_anchor`.
- `FxRangeAccrualOption(initial_date=..., tenor=...)`, the contract's start and length, as on the equity product.
- `BaseFxProduct.record_passed_fixings(fx_env)`: FX theta fixes the fixings its bump steps over (a no-op by default).

### Changed
- `AsianOption.time_shift` (theta) records, at the held spot, a dated record its bump steps over; the bumped price is
  unchanged (pricing filled the same spot) and theta on a fixing day no longer warns `MissingFixingWarning`.
  On a numeric record schedule it now also moves every record's `observation_time` with the maturity (snapped to
  0.0 when zero, as `with_time_elapsed`), so numeric record Asian theta equals the dated twin's (spec fixture, day 3:
  +0.00109532 -> -0.03081166; it used to age the maturity only).
- A fixing at exactly t = 0 (today's, unprinted) that pricing decides at the current spot no longer counts as
  missing: `MissingFixingWarning` names only strictly past fills (`RangeAccrualOption`, `FxRangeAccrualOption` and
  its variants, `AsianOption`; spec 2026-10-09 run fixings §3.2). Prices are unchanged.
- The equity `BacktestEngine` records a range accrual's / record Asian's fixing once, at the spot of the run day it
  was live, when the clock moves past it (`BacktestEngine.run_fixings`), instead of re-deciding it at every later
  day's spot; numeric contracts age with their records (no more constant-maturity "no roll rule" warning for them).
  A run whose instants carry a time of day (`frequency="H"`) and holds such a product is refused. Measured on the
  spec fixture: a dated range accrual on day 10, 2.744374 -> 4.649441 (= the contract with the day-4 fixing recorded).
- `DynamicScenarioEngine` (equity): the same run-fixings recording on a dated path (`DynamicScenarioEngine.run_fixings`);
  registered on the working copy, so the input portfolio is never touched; numeric range accruals / record Asians now
  age on a dated path (they were frozen). An undated path records nothing; a dated path (or an environment) whose
  dates carry a time of day and that holds such a product is refused (R8), and so is a path whose `start_date`
  precedes the environment's valuation date ("the run clock cannot go back", R9).
- A run holding a NUMERIC range accrual / record Asian / FX range accrual that continues past the contract's maturity
  now raises `ValidationError` on the first instant past it, as the dated twin already did (measured: equity
  backtest, equity dynamic scenario, FX backtest); it used to run on with frozen values. Settlement in runs is a
  follow-up.
- `FXBacktestEngine` and `FXDynamicScenarioEngine` record FX range accrual fixings (domestic, foreign, quanto) on the
  run day they pass, at the quoted spot (`run_fixings`); numeric FX range accruals age in FX runs (they were frozen).
  FX range accrual on the spec fixture, day 10: 11,385.19 -> 19,005.46. The FX dynamic scenario registers on its
  working copy (the input portfolio is never touched) and records nothing on an undated path. New refusals
  (`ValidationError`) for an FX run holding an FX range accrual: a path instant carrying a time of day (an FX backtest
  row stamped 15:00, an FX dynamic path whose `start_date` has a time) is refused at its first step (R8), and an FX
  dynamic path whose `start_date` precedes the environment's valuation date is refused ("the run clock cannot go
  back", R9). A book without one runs as before.
- The replay book records a fixing on the date it passes (at that date's spot) and ages a numeric record Asian's
  records with its maturity (spec fixture day 4: numeric 7.807415 -> 7.686469 = the dated twin).
- PnL explain: a range accrual / Asian fixing that passes inside the step (dated t0, or strictly between t0 and t1)
  is decided once at t0's spot for every revaluation of the pair; the t1 snapshot's own decision (its record, or its
  fill at t1's spot) shows in the `fixing_refill` row, now emitted whenever the pair decided a fixing (label: 'fixing
  decided by the pair at t0's spot, decided differently at t1', was 'missing fixing re-filled'). A t1 contract that
  records a fixing the t0 contract left open is no longer refused (a t0-recorded outcome that changes still is), and
  an aged numeric contract passes the roll check (`check_contract_roll(..., pricing_env_t1=...)`). The equity
  `BacktestEngine` explain recorder no longer declares a run-fixings product unrolled (`contract_roll_days=0`): the
  run clock aged it. An equity backtest or replay with `pnl_explain` holding a range accrual / record Asian whose
  fixing passes mid-run explains again (it raised "contract replacement is not a time step"), with a
  `fixing_refill` of exactly 0.0. Spec probe P1: pv_alive_t1 2.744374 -> 4.649441, `fixing_refill` −1.905067,
  waterfall spot −3.126644 -> −1.221578, total unchanged −2.962925.
- A dated contract whose annualized cash needs an accrual start but has no `initial_date` is refused at
  construction (R5); it used to accrue from the valuation date. A contract with no annualized leg is unchanged.
- A `BUSINESS_DAYS`-annualized contract without `accrual_calendar` / `accrual_bus_days_in_year` is refused.
- A numeric schedule with `initial_date` accrues its quoted fraction instead of YF(`initial_date` -> valuation) plus
  the remaining time.
- Periods are derived from the contract's terms when it is built. Changing an accrual term on a built product
  (`initial_date` of a dated contract, a day count, `accrual_factors`, `fixed_coupon_year_fraction`, `tenor`,
  the accrual end, the settlement convention under `SETTLEMENT`, the accrual calendar) needs `product.validate()`,
  which derives the affected periods again (a re-dated record, a new maturity or a new last fixing too); pricing
  before that raises instead of paying the old cash. A schedule set on a built product likewise needs `validate()`.
  `accrued_offset` and `first_period_start` are ageing bookkeeping, read once when periods are derived.
- The intraday twin carries its source records' periods: fixings at different times of day are no longer refused,
  and the twin no longer back-solves `accrual_factors` or banks one `accrued_offset`.
- Monte Carlo REBASED KO-reset reads each path's knock-in anchor for the knock-out accrual and the knocked-in tenor
  (an annualized V1 used to raise).
- An annualized DATED range accrual with neither `initial_date` nor `tenor` is refused at construction, on both
  asset classes; it used to accrue from the valuation date. So is an annualized NUMERIC one booked mid-life (a
  fixing at a negative `observation_time`) without `tenor`: its `maturity` is only the remaining life. Unannualized
  contracts, and numeric ones aged by `time_shift` (which banks the elapsed time), are unaffected.
- A past fixing without its recorded outcome no longer raises. Equity `RangeAccrualOption` and
  `FxRangeAccrualOption` (and the FX foreign / quanto variants) decide a past record without `observed_in_range`
  at the environment's current spot through the product's own range test (per-record barriers, reverse mode); an
  `AsianOption` past record without `observed_price` takes the current spot. One
  `quantark.util.fixings.MissingFixingWarning` (a `UserWarning`) per pricing call names every fixing filled; the
  product is not changed, so pricing equals pricing the same contract with those fixings recorded at the spot. A
  future fixing that carries an outcome is still refused. `resolve_observations(..., warn_missing_fixings=False)`
  skips the warning for a caller that has already warned.
- **An unset `is_annualized_rebate` now follows `is_annualized`** (default True), as the knock-out coupon always
  did; it used to mean a flat rebate. The knock-in payoff is unchanged: an unset `is_annualized_ki` stays flat. An
  explicit True/False always wins. One resolver, `accrual.leg_annualized(product, "ko" | "rebate" | "knock_in")`,
  answers every cash reader (snowball, Phoenix and KO-reset V0/V1, the terminal periods, QUAD V2 terminal
  breaks, the Monte Carlo REBASED V1 tenor, the snowball PDE V0 rows, the KO-reset lifecycle tenors). Every
  factory default contract whose tenor is not one year moves: the V0 rebate is scaled by the tenor. Measured
  (`create_standard_snowball`, rebate = ko_rate 15%, QUAD at spot 100, vol 20%, r 2%): T=0.5 V0 15 -> 7.5, price
  5.413997 -> 3.107317; T=2 V0 15 -> 30, price 0.810672 -> 1.217247; T=3 V0 15 -> 45, price 1.517037 ->
  1.677312; one-year contracts, Phoenix factory defaults (no rebate) and KO-reset schedule 1 of one year are
  unchanged. A certain-V0 contract prices at its closed form (100 + rebate x tenor) e^{-rT} on QUAD, QUAD V2, MC
  and PDE (`test/accrual_time/test_leg_flags.py`). Re-pinned: the replay goldens (13 frames; a 6-day snowball's
  2% rebate falls from 200 to 3.29 per contract: initial value -10131.703047 -> -9974.988063 on `scalar_bsm`,
  see `test/replay_golden/README.md`) and the carry-identity QUAD pins. The modelvalidation snowball builders
  (`equity_snowball.make_snowball`, `equity_snowball_vol.make_snowball`) state `is_annualized_rebate=False`: the
  banked certificates keep the terms they were certified on (user decision 2026-10-09, no re-certification),
  every anchor reproduces exactly, and the test cases built by them (the `pde_statistics` references, the
  first-passage benchmarks) keep their original banks.
- `SnowballOption` no longer refuses an annualized contract with neither `tenor`, `initial_date` nor
  `settlement_date` at construction. R5 is the one rule, read on the effective flags: a dated contract whose
  annualized cash needs an accrual start without `initial_date` is refused (now including one whose rebate
  annualizes by default, e.g. dated by `exercise_date` alone); a numeric contract accrues on its own clock
  (`accrued_offset` + maturity) and is accepted.
- A filled missing past fixing is history (user decision 2026-10-09): it is decided ONCE, at the base valuation's
  spot, and no bump or scenario shock moves it. Only a STRICTLY past fixing is history
  (`quantark.util.fixings.is_history`); one at exactly t = 0 (today, unprinted) is live and moves with spot bumps
  and shocks as before (pricing still decides it at the current spot). `quantark.util.fixings.freeze_missing_fixings(product, env)`
  returns a copy with those fixings recorded (`with_missing_fixings_filled`; `missing_fixing_fills` /
  `with_fixings_recorded` on `RangeAccrualOption`, `AsianOption`, `FxRangeAccrualOption` and its subclasses) and
  warns once; `freeze_portfolio_fixings(portfolio)` does it per position. Applied at: every public
  `GreeksCalculator` method (so `calculate`, bucketed, SIMM and parametric VaR greeks), `BaseEngine.calculate_greeks`
  and `calculate_spot_greeks_curve` (ladder rows keep the base fill, as the engine-grid ladder does),
  `BaseFxEngine.calculate_greeks` (FX calculator, two-rate greeks), `EquityPosition.get_trade_greeks` /
  `get_trade_risk`, the execution-framework spot cells, equity and FX VaR (parametric, historical, Monte Carlo,
  incremental), equity and FX stress tests, SA-CVA (trades and hedges), the surface-shock pipeline (one warning),
  the replay book's carry reprices, and PnL explain. In PnL explain the waterfall's shocked states, the
  alive-at-t1 contract and the Taylor greeks price t0's fill, while `pv_t1` stays the t1 snapshot's OWN value so
  consecutive daily explains chain; a fixing decided differently at t1 shows as a new `Factor.FIXING_REFILL`
  row ("missing fixing re-filled"), present only when t0 had a missing fixing. Three daily explains of a range
  accrual at 109.5 -> 111 -> 111.5 (quantity 2) sum to the PV change exactly (-3.8965), with a re-fill row of
  -3.8054 on the first day. The roll check runs on the contracts as supplied, before any fill. Multi-day runs (equity and FX dynamic scenarios, equity and
  FX backtests, the replay book) fill once at the FIRST valuation and keep it; a fixing that passes during a run
  is recorded once at the spot of the day it was live (run fixings, below). Delta near a barrier no longer carries the
  fill's jump: a numeric range accrual at spot 109.5 under a 110 barrier, delta -0.8914 -> -0.0227 and gamma
  -1.588 -> -0.00106 (equal to the contract with the fixing recorded); an FX range accrual at 1.2995 under 1.30,
  delta -5.906e6 -> -4.991e4. Prices are unchanged (48 prices bit-identical to main); the warning fires once per
  base valuation instead of once per repricing.
- A dated equity option valued ON its exercise date (the same instant) prices its settled value, identical to its
  numeric twin aged to `maturity == 0.0` (user ruling 2026-10-09, option (a)): `get_maturity` returns 0.0 there
  instead of raising `Valuation date (...) must be before exercise date (...)` (`BaseEquityOption`, `OneTouchOption`,
  `DoubleOneTouchOption`, through the new `time_to_exercise`). The settled value is the payoff at the valuation spot
  with every past fixing or event as recorded and a fixing dated the exercise date live (t = 0, fixed at the spot).
  Measured on 16 families x their engines (188 cells, `docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/`):
  on the 154 cells whose numeric twin prices, dated == numeric == the hand settled value to 1e-12, e.g. a range accrual (notional 100, 10%, every fixing in range)
  10.0, a snowball knocked out at 104 15.0. A valuation strictly AFTER the exercise date still raises (now "must be
  on or before exercise date"). Routes with no T = 0 value in either form keep refusing, with `TODO(dated-expiry)`:
  the Heston analytical / MC / PDE and local-vol MC European engines ("maturity must be positive"), the local-vol,
  Heston and SLV barrier MC and Heston / SLV barrier PDE (`t_grid must be strictly increasing`). The replay /
  ensemble dividend-pillar readout on a dated maturity day is unchanged (1e-8).
- QUAD V2 settles on the exercise date (user ruling Q1, 2026-10-10): its compile re-validates its copy with the new
  `validate(allow_zero_maturity=True)` (`BaseEquityOption`, `SnowballOption`, `PhoenixOption`,
  `KnockOutResetSnowballOption`), so a numeric snowball / Phoenix / KO-reset aged to exactly `maturity == 0.0` prices
  its settled value (was `Maturity must be positive, got 0.0`), and so does the dated twin on its exercise date (it
  raised). Snowball KO 103 / continuous KI 75: 15, 15, -30 at spots 104, 90, 70; Phoenix (memory and not): 1, 1, 0,
  -40 at 104, 90, 75, 60. Constructing a contract with maturity 0.0 is still refused, and a negative maturity is
  refused everywhere. The knocked-in KO-reset's numeric twin now settles on V2 (3 at 104, -10 at 90) while the dated
  KO-reset keeps its pre-existing refusal on V2 (a schedule-1 record in the past fails settlement resolution, at any
  valuation date).
- The vol-model European and barrier routes settle on the exercise date (W9, user ruling 2026-10-10): at T = 0 (a
  dated contract on its exercise instant, or the numeric twin at `maturity == 0.0`) each short-circuits before its
  model runs to the expiry code of its Black-Scholes counterpart of the same method, instead of raising `maturity must
  be positive` / `t_grid must be strictly increasing`. Heston analytical, Heston / local-vol / SLV MC European: the
  new `settlement_support.expiry_terminal_value` (the BS analytical and MC engines' expiry branch, now shared);
  Heston / SLV European PDE: `european_expiry_outcome` (the BS European PDE route); local-vol / Heston / SLV barrier
  MC: `barrier_expiry_value` (factored from `BarrierOptionMCEngine._price_expiry_payoff`); Heston / SLV barrier PDE:
  `barrier_expiry_outcome` (the BS barrier PDE route). The nine `TODO(dated-expiry)` markers are gone. European call
  K100: 10 at spot 110, 0 at 90; up-and-out call K100 B130 rebate 5: 10, 0, 5 at 110, 90, 135; a two-day settlement
  lag discounts them (9.99836 at 110). `calculate_greeks` on the exercise date returns the counterpart's keys and
  values (price, delta, gamma; the analytical / MC routes by bump-and-reprice, the PDE routes from the payoff slope)
  where it raised. A valuation after the exercise instant (the next day, or later on the exercise day) and a numeric
  maturity below 0 still refuse. Not changed: a knock-in carried as triggered (`_otc_lifecycle_knocked_in`) is not
  read at T = 0 by any Black-Scholes barrier route (up-and-in call B130 rebate 5 at spot 110: the rebate 5, not the
  payoff 10), so the vol-model barrier routes now return the same (open question in the W9 report; fixed below,
  "A knock-in carried as triggered decides the exercise date"). Evidence:
  `docs/superpowers/specs/2026-10-10-volmodel-expiry-settles-evidence/`.
- A numeric run-fixings participant (range accrual, record Asian) is no longer floored at a 1e-8 maturity in the
  replay (run-fixings plan Ruling 12, superseded now the dated twin settles on its exercise date): on its maturity
  day it is exactly 0.0 and prices its dated twin's settled value (numeric Asian pricing copy on T: 6.9999999965 ->
  7.0, = the dated twin); past it the maturity is negative and pricing refuses, as the dated twin past its date
  does. The replay dividend-pillar readout maps 0.0 / negative to the same 1e-8 pillar (unchanged). PnL explain's
  `check_contract_roll` expects a participant's `with_time_elapsed` maturity (snapped, not floored), so an explain
  step ending on its maturity explains, equal to the dated twin (it raised `alive product maturity 0.0 is not ...
  rolled ...`). A backtest of the twins through T values both at the settled value on T; a run to T + 1 is refused
  for both, as before.
- Asian MC at maturity 0.0 averages the recorded fixings and the live one at the spot (as the analytical engine
  does) instead of the spot alone: 11 fixings 101..111 recorded, spot 110, strike 100: 10.0 -> 6.3333.
- Accumulator analytical and MC at maturity 0.0 settle the observation at the maturity instant on the spot (as at
  every other instant, `settle_valuation_instant`) instead of dropping it: spot 100, strike 95, one share: 0 -> 5;
  `test_accumulator_mc_engine` expiry cases 5.0 -> 10.0 and -5.0 -> -15.0.
- The DCN settles on its final observation date (W10, user rulings 2026-10-10 on
  `docs/superpowers/specs/2026-10-10-dcn-final-observation-proposal.md`): every DCN route (GBM, local-vol, Heston,
  QE and coupled-coarse MC, the execution-session MC adapters, the GBM, local-vol and Heston PDE) prices the settled
  final row from the new neutral `quantark.asset.equity.engine.dcn_valuation` (`settle_final_observation`) instead of
  raising `DCN grid has no steps`, `tau must be positive` or `t_grid must be a 1D array`. The row is decided on
  today's spot, pointwise at barrier ties on every route: knock-in = carried OR spot <= KI; knock-out (spot >= KO,
  either state) pays the KO coupon; the fixed coupon pays when not knocked out, not knocked in and spot >= the
  coupon barrier; the loss leg pays when knocked in and not knocked out; each flow at its own payment-date DF (1 when
  paid that day). DCN_A (N 1e6, S0 6000, lags 2 SSE days): 9996.099391 for a coupon or knock-out, -433164.306946
  knocked in at 4000; the MC result's legs / event statistics are the 0/1 decisions with `std_error` and
  `expected_life_years` 0.0; PDE greeks are the settled row's slope (0 at a tie), gamma 0. Sessions answer the
  final date natively (the kernel's new optional adapter hook `answers_natively(prepared)`: no batch plan). The
  cash-Greeks report's date-rolled `theta_1d` on the penultimate day now prices (settled - today) where it raised.
  The row equals the MC path kernel's one-column row bit for bit and is the limit of both engines as the last step
  shrinks (evidence: `docs/superpowers/specs/2026-10-10-dcn-final-observation-evidence/`). Not changed: decided but
  unpaid coupons are not carried (W14), a knocked-out DCN is assumed alive (W14), `settlement_convention` is ignored
  (W15), a valuation after the final date still raises.

### Fixed
- The autocallable MC time grid holds one node per instant (W17a, approved 2026-10-10). The snowball, KO-reset and
  Phoenix MC grids (`SnowballMCEngine._build_time_grid` / `_build_time_grid_ko_reset`, `PhoenixMCEngine.
  _build_time_grid`, inherited by the local-vol / Heston / QE / SLV snowball and Phoenix MC) were the sorted Python
  `set` of their times, so one instant spelled two ways an ulp apart -- a knock-out record against a continuous-KI
  `np.linspace` node (a dated record's `(d - a) / 365`, an aged numeric one's `d / 365 - a / 365`, `T * j / n`),
  a REBASED `t_ki + offset`, or KI dates rounded to 10 decimals -- survived as two nodes ~1e-16 years apart. The
  empty step carries no variance but adds a column of random draws and re-deals every draw after it, so a dated
  contract and its aged numeric twin priced on two independent samples (W17 diagnosis: day 92 of a 256-path snowball
  12.08 vs 11.34; at 1M paths the gap is within one standard error, both forms agree with the PDE / QUAD V2).
  New `quantark.util.numerical.merge_time_nodes(times, tol=Tolerance.ZERO, pin=T)` collapses every run of nodes
  closer than `tol` into its LAST member (a maturity within `tol` is pinned back to exactly `T`) and `node_index`
  maps a time to the first merged node strictly after `t - tol`; the three grid builders use them. A grid with no
  such run is bit-identical. Moved: only MC values whose grid held such a run, each a fresh draw of the same law --
  on the W12 snowball / Phoenix, 6 dated and 8 numeric valuation days of 364 (e.g. BS snowball day 92 numeric,
  256 paths seed 7: 10.756027 -> 11.461036 = the dated value), the continuous-KI KO-reset on days 26 / 65 / 78 / 82,
  `test_ko_reset_event_stats_shape[Rebased]` pv 98.1788 -> 97.5565 (shape-only test) and the live gate
  `test/pde_grid/test_anchor_certification.py::test_snowball_pde_vs_mc[False]` MC 96.645369 -> 96.621477 (|PDE - MC|
  0.0213 -> 0.0026, gate 0.2). No golden or certificate anchor moved (probe over `test/modelvalidation`,
  `test/mo_volmodels`, `test/pde_grid` and every autocallable-MC test file). Dated == numeric twin now within
  8.9e-15 on every day and route measured. The range-accrual and Asian MC grids use the same set union but hold no
  such pair (only observation times against `T`, spelled alike in both forms; every day measured): unchanged. New
  `test/test_mc_time_node_merge.py`. Evidence: `docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis/`
  (`moved_values.txt`).
- The local-vol, Heston and SLV snowball MC price a KO-reset with a CONTINUOUS knock-in (W17 side ticket). The
  KO-reset path hands the Brownian bridge its paths cut at the knock-in horizon (schedule 1) while these engines
  record the per-step log-variance over the whole grid, so the shape check raised `PricingError` ("recorded step
  variance ... does not describe these paths") on every such contract. The caller now names the horizon
  (`horizon_steps`) and the record is sliced to it; any other mismatch still raises. Flat-IV local-vol MC agrees with
  the BS MC within 4 standard errors; dated == numeric twin within 1e-10. Moved values: none (raise -> price). New
  `test/test_ko_reset_vol_mc_continuous_ki.py`.
- A dated autocallable whose discrete knock-in records lie in the past prices like its numeric twin (W11). The
  knock-in resolvers (`SnowballOption.resolve_ki_observations`, inherited by `KnockOutResetSnowballOption`, and
  `PhoenixOption.resolve_ki_observations`) resolved EVERY record, so a dated record before the valuation date asked
  the settlement resolver for a year fraction from the valuation date back to it (`calculate_year_fraction`,
  `day_counter.py:86`: "end_date (2025-02-04) must be after start_date (...)"), while the knock-out resolvers already
  kept only the records not before the valuation date and the numeric twin's `time_shift` had dropped them. Every
  engine reads that resolver (QUAD V2's compile, the PDE's valuation decision and knock-in rows, QUAD V1's valuation
  decision, the MC time grid), so a dated KO-reset with schedule-1 records in the past was refused mid-life on all of
  them (QUAD V2 and the PDE also when knocked in, QUAD V2 also on the exercise date), and so was a dated snowball or
  Phoenix with a DISCRETE knock-in schedule (the continuous-KI fixtures never met it). The knock-in resolvers now keep
  the same records as the knock-out ones (`observation_schedule.active_record_indices`; new
  `active_ki_source_indices` on both products): a passed record is history, its outcome the carried knock-in state,
  never resolved forward. The KO-reset MC grid maps each kept record to its source position in the contract's
  schedule (the knock-in anchor's `source_index`). The KO-reset contract rules are unchanged (a past schedule 1 not
  knocked in still refuses, `require_alive`, on both forms). Moved: 363 dated values raised -> price / delta / gamma
  (KO-reset ABSOLUTE at days 100-365 alive and knocked in on MC / PDE / QUAD V1 / V2 / local-vol / Heston / SLV MC,
  e.g. QUAD V2 292 days in knocked in at spot 100: raised -> 2.43400 = the numeric twin; QUAD V2 on the exercise date
  3.0 at 104, -10.0 at 90; discrete-KI snowball and Phoenix on MC / PDE / QUAD V1 / V2 / local-vol MC and PDE /
  Heston / QE / SLV MC); no numeric value and no golden / certificate anchor moved. Dated == numeric twin (price,
  delta, gamma) within 7e-13. Intraday event building (`quantark/intraday/events.py`) now refuses a schedule env that
  drops a knock-in record, as it already refused one that drops a knock-out record (it indexes the source records by
  position). Not changed: the 2D Heston / SLV snowball and Phoenix PDE still refuse a dated knock-out schedule before
  maturity (pre-existing, a different step). New `test/test_ko_reset_past_records.py`; the pinned
  splits are gone (`test_dated_expiry_settles` `DATED_REFUSED_ANY_DATE`, `test_expiry_carried_knock_in` KO-reset
  entries in `AUTOCALL_REFUSED` / `AUTOCALL_BEFORE_REFUSED`). Evidence:
  `docs/superpowers/specs/2026-10-10-ko-reset-past-records-evidence/`.
- `EuropeanMCEngine`'s no-arbitrage lower-bound check is statistical (W13, user decision 2026-10-10; diagnosis
  `docs/superpowers/specs/2026-10-10-mc-lower-bound-diagnosis/`). A deep in-the-money European near expiry is worth
  its discounted intrinsic bound (Black-Scholes minus the bound is 0.0 in double precision one day out), so the
  unbiased estimate fell below the bound about half the time and the hard `price < bound - 1e-6` check refused it
  (38-62 % of seeds at every path count and method; no estimator bias measured, |bias| < 3e-6). The engine now
  refuses only an estimate more than `max(Tolerance.PRECISION, 4 * std_error)` below the bound
  (`EuropeanMCEngine.LOWER_BOUND_STD_ERRORS = 4.0`; the std error it reports, in contract units) and keeps the hard
  `Tolerance.PRECISION` (1e-6) floor when the std error is zero or undefined. The estimator is unchanged: every price
  that passed before is bit-identical (1162 of 1162 grid cells); only refused cells now price (182 of 182, raised ->
  their own estimate; `moved_values.txt`). The BS barrier MC prices the day-before carried knock-in down-in put
  through this engine: 9.993453009252772 at spot 90 and 24.993174367657804 at spot 75 (dated form; the numeric twin
  differs in the last digits), so `("mc", "down_in_put")` leaves `_BEFORE_REFUSED` in
  `test_expiry_carried_knock_in.py` and its day-before continuity assertion runs. Under antithetic and plain QMC the
  reported std error is the iid formula and overstates the real error (up to about 800x measured), so the check is
  loose there; the std error formula, `EuropeanQuadEngine`'s check and RQMC seeding are unchanged (open questions in
  the diagnosis).
- Antithetic MC runs report the pair-mean standard error (W16, signed W13 follow-up 1, spec I3/A9). Within each batch
  the samples are `[z; -z][:n]`; with `m = ceil(n/2)` the pairs are `(i, m+i)`, an odd `n` leaves the unpaired
  `z[m-1]` out of the s.e. only, the pair means of all batches pool into one unit set, and the s.e. is
  `std(units, ddof=1)/sqrt(#units)`. New `quantark.montecarlo.std_error` (`antithetic_pair_means`,
  `mean_and_std_error`, `sample_std_error`) is used by the European, digital, Asian, range-accrual, American, barrier
  (continuous, discrete, expiry), accumulator, single and double sharkfin MC engines, the vol-model (local vol,
  Heston, QE, Heston-SLV) snowball and Phoenix engines, the DCN GBM / local-vol / Heston engines and their execution
  adapters, and the FX barrier, sharkfin, range-accrual, TARF and TARN MC engines. Plain QMC keeps the iid formula,
  now documented as not a valid error bar for deterministic QMC (use `RANDOMIZED_QUASI`). The plain GBM snowball and
  Phoenix engines draw iid paths whatever `use_antithetic` says and keep the iid s.e.; the Dask batch route of the
  snowball and Phoenix engines reduces sum / sum-of-squares and keeps the iid s.e. (`# TODO(w16)`). The W13a lower-bound
  check inherits the corrected s.e.: no refusal flips (0 of 300 seeds refused at spots 90 and 75, 1 and 5 days, 256
  antithetic paths, before and after; the one-day deep-ITM put s.e. falls 0.0636 to 0.0007). Moved values: reported
  standard errors only (examples, 2000 antithetic paths, fixed seed: European call 0.3435 -> 0.2671, cash-or-nothing
  0.1085 -> 0.0193, American put 0.1991 -> 0.1115, FX TARF 0.00722 -> 0.00052, DCN GBM 3220 -> 2680, Phoenix local-vol
  0.01441 -> 0.01022; range accrual rises 353 -> 497 because its payoff is even in z and the iid formula understated
  it); prices bit-identical. New `test/test_mc_std_error_antithetic.py`.
- RQMC seed spacing is documented (W16, signed W13 follow-up 3): batch `b` scrambles with `base_seed + b`, so
  replications whose base seeds sit closer than their batch count share scrambles. `SobolNormalGenerator._make_engine`
  and `docs/execution/policy-guide.md` ("RQMC seeds and replications") state the rule: space base seeds by at least
  the batch count, or derive them from a hash / `SeedSequence`. Seeding unchanged. Moved values: none.
- The DCN PDE routes pay a monthly observation that falls on the valuation date (W10b, user ruling Q5 2026-10-10).
  The GBM, local-vol and Heston DCN PDE dropped it (the march never applies t = 0) while every MC route paid it, so
  the PDE value lost the day's coupon overnight. They now decide the row pointwise at the spot through the W10
  helper (`dcn_valuation.observe_valuation_row`), at every valuation date: a knock-out today settles at the
  knock-out coupon (greeks 0, no march), a coupon today is added to the smooth continuation, a knock-in today reads
  the knocked-in surface (as before). DCN_A on 2024-12-03 (month 23): spot 5500 coupon today 9779.74 -> 19777.79
  (MC 19767.09; the day before 19739.93); carried knock-in at 6300, knock-out today -23212.00 -> 9998.05 (= MC).
  The grid spot curve (`calculate_spot_greeks_curve`, `BranchSolution`) folds the same row per level. 90 values
  move on the measured observation dates; nothing else (evidence `moved_values.txt`).
- The vol-model routes resolve DATED observation schedules (W12, user-approved). (1) The Heston and SLV snowball and
  Phoenix PDE (2D ADI) refused a dated knock-out schedule at any date before maturity ("a date-based knock-out
  observation schedule needs a pricing environment to resolve its observation times"): their S-axis grid request
  (`_layer_x_nodes`, and the bump context's frozen layout) never received the environment the 1D adapter stashes.
  Both now call `_stash_grid_request_env` before `grid_request`, as the 1D adapter does. (2) The local-vol, Heston and
  SLV barrier MC and the Heston and SLV barrier PDE read only the legacy `observation_dates`, which a dated schedule
  leaves unset ("discrete monitoring requires observation_dates"). A dated schedule is now resolved against the
  environment exactly as the Black-Scholes barrier MC resolves it (`schedule.resolve(env, default_barrier=,
  default_payoff=, require_single=True, product=)`, new `engine/vol_barrier_schedule.discrete_observation_times`)
  and the resolved times feed the kernels; a numeric schedule is still read from `observation_dates` verbatim. The
  limits are unchanged and identical for both forms: per-observation varying barriers / payoffs, an observation
  after maturity, a record behind the valuation date (refused by the resolution, as on the Black-Scholes routes),
  a delayed pay-at-hit rebate, a memory Phoenix on the 2D PDE. Moved values: dated forms only, raised -> price;
  every numeric value is unmoved (PDE byte book and the modelvalidation anchors bit-identical). The dated twin
  equals the numeric twin aged to the same date (153 of 201 measured values bitwise, worst 3.5e-12 relative: the
  last ulp of the resolved times). New `test/test_vol_dated_schedules.py`; the W9 / carried-KI pins on these
  refusals are removed (`_BEFORE_REFUSED`, `AUTOCALL_BEFORE_REFUSED` in `test/test_expiry_carried_knock_in.py`).
  Evidence: `docs/superpowers/specs/2026-10-10-vol-dated-schedules-evidence/`.
- The vol-model barrier routes price a discrete schedule at its records' own terms, first-hit only (W18, user rulings
  2026-10-10). The local-vol, Heston and SLV barrier MC and the Heston and SLV barrier PDE accept a discrete schedule
  whose records all share one level and payoff, but handed the kernels `product.barrier` and `product.rebate`: a
  schedule whose common level differs from `product.barrier` was priced at the wrong level, silently, numeric and
  dated alike; so was a knock-out's common record payoff. The kernels now receive the records' common level (a
  record without a level takes `product.barrier`, the product's own normalisation) and, on a knock-out, the
  records' common payoff, exactly as the Black-Scholes barrier MC reads them (`_build_discrete_grid`); a knock-in's
  no-hit rebate stays `product.rebate`, as the Black-Scholes MC pays it. New
  `engine/vol_barrier_schedule.schedule_barrier_terms`. The vol kernels monitor first-hit only: a schedule with
  `aggregation_mode` ACCUMULATE, BEST or WORST, which they silently priced as STOP_FIRST_HIT, is now refused
  (`ValidationError`, "aggregation ... is refused"; new `refuse_non_first_hit`, `# TODO(vol-aggregation)`), on both
  forms and including the Heston / SLV PDE deterministic-variance limit. CONTINUOUS / EXPIRY monitoring is
  unaffected. Moved values: level / payoff-differing schedules value -> value (each equal, bitwise, to the contract
  booked at the records' terms), non-first-hit aggregation value -> refused; every schedule at `product.barrier`
  with STOP_FIRST_HIT is unmoved (PDE byte book and the modelvalidation anchors bit-identical). New
  `test/test_vol_barrier_schedule_level.py`. Evidence: `docs/superpowers/specs/2026-10-10-vol-barrier-schedule-level-evidence/`.
- The Black-Scholes barrier MC decides at the valuation instant on the schedule record's own level (W18b, user
  approval 2026-10-10). Two decisions of `BarrierOptionMCEngine` read `product.barrier` / `product.rebate` while
  its discrete path reads each resolved record: (a) the exercise-date value `barrier_expiry_value` (also the
  local-vol / Heston / SLV barrier MC routes' value at T = 0 since W9) and (b) the "hit at valuation" decision
  before simulating. For a DISCRETE schedule both now read the first record at or after the valuation instant (the
  record on the valuation date, else the next observation, the barrier PDE's `_next_observation` rule): its level,
  and on a knock-out its payoff (new `valuation_barrier_terms`; `barrier_expiry_value` now takes the pricing
  environment). A record without a level takes `product.barrier`; CONTINUOUS / EXPIRY monitoring and a knock-in's
  no-hit rebate are unchanged. Example: up-and-out call, `barrier` 120, record at 115 on the exercise date, spot 117:
  17.0 -> 5.0 on the BS MC and the three vol MC routes, equal to the BS PDE, the vol PDE routes and the contract
  booked at 115. A record on a mid-life valuation date whose level the spot has crossed now knocks out / in at once,
  as the PDE does (the MC previously simulated and failed on the zero-length step). Moved values: only contracts
  whose record level differs from `product.barrier` (or a knock-out record paying other than the rebate), spot
  between the levels; plain contracts bit-identical. New `test/test_barrier_mc_record_level_at_valuation.py`.
- The Black-Scholes barrier MC prices a DISCRETE schedule with a record on the valuation instant that the spot has
  NOT crossed (it raised "dt_array must have strictly positive entries"). Such a record is decided "not hit" at the
  valuation instant (W18b) and leaves the simulated grid: the price equals, bitwise, the contract aged past that
  record; a sole such record leaves the never-hit value (knock-out: the vanilla; knock-in: the rebate at the
  terminal). Moved values: none (raise -> price).
- A knock-in whose discrete schedule records pay other than `product.rebate` is refused on every route (W18b, user
  ruling 2026-10-10, option (a)). A knock-in's rebate is paid when no observation hits, so no record's payoff can
  apply. The 1D PDE core already refused it (spec 3C §5.2 (a): the Black-Scholes and local-vol barrier PDE, and the
  Heston / SLV PDE deterministic limit); the Black-Scholes MC, the QUAD barrier, the local-vol / Heston / SLV MC and
  the Heston / SLV 2D PDE priced it. The core's rule and message ("a knock-in has one rebate (R); a schedule record
  pays P (spec 3C §5.2 (a))") move to the neutral `engine/barrier_schedule_rules.refuse_knock_in_record_payoff`,
  which every route now calls, in both forms. Unaffected: knock-ins whose records pay the rebate or nothing, and all
  knock-outs. Moved values: those knock-in cells price -> refused only. New
  `test/test_knock_in_record_payoff_refused.py`.
- The analytical barrier routes read the schedule record's level and payoff on a DISCRETE schedule (W18c, spec
  2026-10-13 I4 / A8): `BarrierAnalyticalEngine`, `OneTouchAnalyticalEngine`, `DoubleBarrierOptionAnalyticalEngine`
  and the single / double sharkfin analytical engines read `product.barrier` (`upper_barrier` / `lower_barrier`) in
  their hit pre-checks, immediate knock-out cash and barrier-shift formula. Now a DISCRETE schedule is hit now only by
  a record at the valuation instant, at its own level(s), and an immediate knock-out pays that record's payoff
  (`valuation_barrier_terms`, moved to the neutral `engine/barrier_schedule_rules.py` and still importable from the
  MC module, plus the new `valuation_double_barrier_terms`); the forward closed form (shift, beyond-the-barrier
  first-observation integral, one-touch volatility) uses the level(s) every remaining record (t > 0) shares
  (`forward_barrier_level` / `forward_double_barrier_levels`). Remaining records with different levels have no
  closed form: `ObservationSchedule.assert_analytical_ready` (new `pricing_env` argument, `remaining_records`,
  `remaining_levels`) raises `ValidationError`, as it does for mixed payoffs; a record on the valuation instant still
  decides there. A future knock-out hit pays the hit record's payoff too (the MC's `payoffs[first_idx]`, the PDE):
  the forward closed form and the first-observation hit value use the remaining records' payoff
  (`forward_knock_out_cash`; one fixed payoff is already required, an unset payoff takes `rebate` /
  `knock_out_rebate`); knock-ins and no-touches keep their rebate. The sharkfins' no-rebate knock-out legs no longer
  carry the record payoffs (their cash legs pay them), and a single sharkfin with `knock_out_rebate` 0 whose records
  pay cash no longer skips its touch leg. CONTINUOUS / EXPIRY monitoring unchanged. Moved values (flat 20% vol, r 3%, q 1%, K 100, T 1,
  monthly records; "now" = aged to the 3-month record): up-and-out call `barrier` 120, records 115, spot 110:
  4.14526 -> 3.95226 (up-and-in 16.16050 -> 16.35350); same, now, spot 117: 4.79158 -> 4.88876 (records paying
  4.5: -> 4.39988); one-touch up 120 / records 115, spot 110: 2.65102 -> 3.37817, now at 117: 3.58628 -> 4.88876;
  double knock-out 130/75, records 125/80, spot 100: 4.52666 -> 3.68421, now at 127: 4.07797 -> 2.0; single
  sharkfin 120 / 115, spot 110: 3.52510 -> 2.89580, now at 117: 3.61756 -> 2.93325; double sharkfin 130/75 /
  125/80, spot 100: 5.49710 -> 4.65465, now at 127: 5.05572 -> 2.93325 -- each equal, bitwise, to the contract
  booked at the records' levels; mixed remaining levels value -> refused. Records paying 4.5 (product cash 5, a
  sharkfin's 3), spot 110 / 100: up-and-out call records at 120: 4.14526 -> 3.88016, records at 115: 4.14526 ->
  3.61444; one-touch: 2.65102 -> 2.38591; double knock-out (rebate 2): 4.52666 -> 5.12593; single sharkfin: 3.52510
  -> 4.32040; double sharkfin: 5.49710 -> 5.85667 -- each equal, bitwise, to the contract booked with cash 4.5.
  Single sharkfin `knock_out_rebate` 0, records paying 2.5: 1.93449 -> 3.25999; records at 115, now at 117: 1.46580
  -> 2.44438 (2.5 at the terminal). Plain contracts (records at `product.barrier`, paying the product's cash)
  bit-identical. New `test/test_barrier_schedule_level_analytical_quad.py`.
  The QUAD barrier, one-touch and double-barrier adapters (`quad_adapters.py`) decide at the valuation instant as
  the Black-Scholes barrier MC does (W18b): the immediate hit / touch and the expired branch read the first record at
  or after the valuation instant (`valuation_barrier_terms` / `valuation_double_barrier_terms`), its level(s), and a
  knock-out (one-touch) pays its payoff; the forward lattice was already record-aware and prices mixed levels natively
  (no refusal). The knock-out / one-touch rebate lattice is no longer skipped when the product's own cash is 0 but the
  records pay cash. Moved values (QUAD 401 points; up-and-out call `barrier` 120 / records 115, K 100): spot 117 beyond
  the next record 4.50422 -> 4.85223 (5 at the terminal), records paying 4.5 aged onto a record at 117: raised "t_grid
  must be strictly increasing" -> 4.39988, up-and-in at 117: 21.32214 -> 20.97413 (the vanilla), expired at 117: 17.0
  -> 5.0; rebate 0 with records paying 4.5, spot 110: 1.40591 -> 3.81059; one-touch 120 / records 115 at 117: 4.28985
  -> 4.85223, expired at 117: 0.0 -> 5.0, rebate 0 records paying 4.5: 0.0 -> 2.40468; double knock-out 130/75 /
  records 125/80 at 127: 2.48798 -> 1.94089, expired at 127: 27.0 -> 2.0 -- each equal, bitwise, to the contract
  booked at the records' terms and to the BS MC decision. A knock-in whose records pay other than `product.rebate` is
  now refused on the analytical barrier and double-barrier engines and on the QUAD double barrier too (Ruling R19,
  `refuse_knock_in_record_payoff`, the message every route raises), making the W18 Q3 "every route" rule true: e.g.
  analytical up-and-in, records paying 4.5 (rebate 5): 16.16050 -> refused; double knock-in, records 1.5 (rebate 2):
  analytical 6.24155 / QUAD 5.91733 -> refused. QUAD plain schedules and EXPIRY monitoring bit-identical.
  QUAD prices a DISCRETE schedule with a record on the valuation instant that the spot has NOT crossed (it raised
  "t_grid must be strictly increasing" on the barrier, one-touch, double-barrier and double one-touch adapters, plain
  contracts included): the record is decided "not hit" there and leaves the lattice (I5 / A14, as the BS barrier MC)
  -- bitwise the contract aged past it; a sole such record leaves the never-hit value (the remaining-contract
  replication, as once monitoring has ended); two records on that instant stay refused. The double one-touch /
  no-touch QUAD adapter decides at the valuation instant on the deciding record too (`valuation_double_barrier_terms`;
  a double one-touch pays its payoff; its rebate lattice no longer skipped when only the records pay). No analytical
  engine prices `DoubleOneTouchOption` (`analytical/double_no_touch.py` is a probability function on explicit
  levels), so there is nothing to fix there. Moved values (QUAD 401 points, aged onto the 3-month record, spot inside):
  raised -> priced, e.g. up-and-out call 120: 4.37956, up-and-in: 14.74546, one-touch: 2.40644, no-touch: 2.88655,
  double knock-out: 5.57745, double one-touch: 0.77959, double no-touch: 4.10917; sole record: knock-out 14.23864
  (the vanilla), knock-in / no-touch 4.88876 (the rebate at the terminal), touches 0.0, double knock-out 7.56296.
  Double one-touch 130/75, records 125/80: spot 127 (beyond the next record) 4.31809 -> 4.85223 (hit-paid, at 77:
  4.64745 -> 5.0), expired at 127 0.0 -> 5.0, records paying 2.5 aged onto a record: raised -> 2.44438; rebate 0 with
  records paying 4.5: 0.0 -> 1.77979; double no-touch at 127: 0.53413 -> 0.0, expired 5.0 -> 0.0. Plain double
  one-touch / no-touch (28 cells) bit-identical.
- A knock-in carried as triggered decides the exercise date (user ruling 2026-10-10, "check knock in state first").
  At T = 0 (a dated contract on its exercise instant, or the numeric twin at `maturity == 0.0`) the barrier engines
  decided a knock-in from today's spot / records only and paid the rebate to a contract carried as knocked in
  (`lifecycle_state.knocked_in` or the product's `_otc_lifecycle_knocked_in`), while the day before the same state
  priced the knocked-in vanilla. The knock-in decision at T = 0 is now the carried state first
  (`monitoring_ended.knock_in_carried`, the helper the T > 0 path reads), then today's observation: knocked in pays
  the payoff, only a contract never knocked in the rebate. Fixed in the shared expiry code, so every route gets it:
  `BarrierAnalyticalEngine._price_at_maturity`; `barrier_expiry_value` (BS barrier MC, and the local-vol / Heston /
  SLV barrier MC routes); the PDE knock-in adapters' `maturity_value` (BS and local-vol barrier PDE, the Heston / SLV
  barrier PDE routes through `barrier_expiry_outcome`, double barrier PDE; greeks: delta the payoff slope); the QUAD
  barrier and double barrier expired price; `DoubleBarrierOptionAnalyticalEngine._price_zero_maturity`. Up-and-in
  call K100 B130 rebate 5 carried as knocked in: 5.0 -> 10.0 at spot 110, 5.0 -> 0.0 at 90 (both forms, every
  route); down-and-in put B80: 5.0 -> 10.0 at 90, 5.0 -> 0.0 at 110; double knock-in 80/130: 5.0 -> 10.0 at 110.
  The numeric twins moved from the same wrong values on main. A knock-in not carried, a carried knock-out / touch
  (terminal lifecycle state, read before maturity on both sides of T = 0) and every autocallable with a knock-in
  term (snowball, Phoenix, KO-reset, on MC / PDE / QUAD V1 / V2 / local vol / Heston / SLV, which already read the
  carried state at T = 0) are unchanged. New `test/test_expiry_carried_knock_in.py` (hand settled values, both
  forms, both carriers, day-before continuity); W9's `up_in_call_knocked_in` cells are now also pinned to the hand
  value. Evidence: `docs/superpowers/specs/2026-10-10-carried-knock-in-at-expiry-evidence/`.
- Snowball / Phoenix / KO-reset PDE (1D BSM and local vol, 2D Heston and SLV): the not-knocked-in (V0) Dirichlet
  rows are built by edge. The knock-out edge is the upper row on a standard product and the LOWER row on a reverse
  one; it carries the knock-out payoff once it lies beyond every knock-out level. A reverse product used to price
  its upper (knock-in side) row as the knock-out payoff and its lower (knock-out side) row as principal + rebate.
  A call-rebate contract's no-event rows pay principal + the participation's forward intrinsic value (a put on a
  reverse Phoenix) and no fixed rebate; the lower row used to add `rebate_rate x S0` (the factory sets it to the KO
  rate). Under continuous knock-in the knock-in edge copies the knocked-in row only when it lies at or beyond the
  knock-in level; it used to copy it unconditionally, so a knock-in barrier the grid never reaches still knocked in
  the edge. Certain-V0 controls (`docs/superpowers/specs/2026-10-08-accrual-scope-followups-evidence/
  item7_v0_rows_by_edge.py`, PDE - closed form, flat / annualized): reverse fixed rebate +1.1e-3 / +5.0e-4 ->
  6.7e-7 / 6.6e-7; standard call rebate with `rebate_rate` 0.10 +3.4e-4 / +3.5e-4 -> 4.0e-6 / 7.2e-6; reverse call
  rebate -9.2e-4 / -3.6e-3 -> 6.7e-6 / 1.3e-5; continuous KI off the grid -2.7e-3 / -3.0e-3 -> 2.4e-6 / 2.5e-6.
  The earlier "cancellation" on the call-rebate rows was QUAD's: on its default 1001-point grid QUAD overprices the
  call leg by 4.4e-4 / 8.7e-4 against the closed form (2001 points 1.4e-4, 4001 1.4e-5, 8001 7.2e-6), so PDE - QUAD
  read better with the phantom rebate. Every banked certificate anchor, `pde_statistics` reference and replay golden
  reproduces bit for bit (the cert reverse and call-rebate cells' edges lie too far from the spot to reach the price),
  and so does the PDE byte book (`test/pde_bitwise/capture_book.py`, 713 rows, 33201993 vs this branch, same machine:
  0 diffs; the same 10 rows raise on both sides).
- Snowball / Phoenix PDE, 2D Heston and SLV: under continuous knock-in, the not-knocked-in edge that lies beyond the
  knock-in level copies the knocked-in surface's row of the same step. The 2D hooks solve V1 to completion before V0
  and kept no `_grid_v1`, so the edge row was 0.0 (standard lower edge, KO 103 / KI 75: 0.0 -> 10.2004; reverse
  upper edge, KO 97 / KI 125: 0.0 -> -791.999). They now compute V1's rows first, as the 1D adapter does
  (`SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows`). Prices move at round-off (|move| <= 4e-13) on
  the contracts measured, and by up to 8.1e-5 only when the knock-in level sits just inside the grid's lower edge.
  Discrete knock-in is untouched (`docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/`).
- 2D Heston / SLV ADI core (`HestonSLVADICore._A1`, `v0_boundary="degenerate_pde"`, the 2D snowball / Phoenix / DCN
  default): the explicit S-operator now carries the degenerate v = 0 column, `(r - q) U_x - r U`. The implicit
  S-solve already carried the S-drift and the -rU reaction on every variance column, so the v = 0 row evolved under
  theta x A1, consistent only on a fully implicit step (W7 ruling Q1, 2026-10-10). The standard certain-V0 control
  under Heston at (96, 24, 48), every damping step off, goes from +2.478e-3 to -9.31e-7 against its closed form; with
  the default damping +3.827e-3 -> +3.761e-3 (the rest is the damped steps' dt^2 term, by design: ruling Q4).
  Every 2D Heston / SLV price on that boundary moves: `adi2d-snowball-greeks` re-certified by amendment
  (2026-10-10; 28 of 28 anchors moved, low_feller delta +2.66e-4 / +2.57e-4 the largest; both candidates ADMITTED,
  the low_feller errors against the banked reference shrink); the reverse continuous-KI Heston Phoenix (KO 97 / KI
  125) at (192, 48, 192) 95.7019 -> 95.5960, now monotone in refinement; the Feller-violating Heston DCN at (151, 41)
  -89176.7 -> -90111.4 per 1,000,000 notional. Values: `docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/moved_values.txt`.
- Phoenix PDE (1D, and the 1D statistics march of the 2D Heston / SLV Phoenix): under continuous knock-in the
  not-knocked-in edge beyond the knock-in level copies the knocked-in row of the step; `PhoenixAdapter` marched
  every V0 memory state before the knocked-in surfaces and never exposed them, so the row was 0.0 (W7 ruling Q2). A
  V0 group now sets V1's rows of the step first and exposes them for that row, as the snowball adapter does, and the
  Phoenix statistics copy the knocked-in row too (`PhoenixStatistics._alive_ki_edge_copies_ki = True`). Moves short
  continuous-KI Phoenixes whose edge sits near the knock-in level (three months, reverse KI 150: +1.737e-5);
  `phoenix-flat-bsm` re-certified by amendment (2026-10-10; near_expiry pv -2.8e-16, delta +6.9e-17, the other 124
  anchors bit for bit).
- Snowball / Phoenix PDE continuous knock-in on an up (reverse) barrier, FIRST_PASSAGE crossing correction
  (`_ContinuousKIFirstPassage._geometry`, W7b, ruling 2026-10-10). The half-cell floor on the slope sample measured
  the cell at the grid's far edge on an up barrier (a negative "cell": the floor never acted), so a node a few
  thousandths of a cell below the barrier sent the 2D Heston / SLV price off by ~x8 per step; and the reflection
  factor overflowed to inf x 0 = NaN on the 2D v = 0 column. The floor now measures the cell at the barrier, and the
  factor is evaluated in one exponent where it would overflow (`Tolerance.EXP_MAX`). Fixed: reverse KO 97 / KI 125 at
  n_x 312 / 317 / 384 (diverged to -1e30 ... -1e173), the production Greek grid at KI 139.80 (raised `PricingError`),
  the production PV grid at KI 114.45 (NaN); other reverse 2D continuous-KI grids whose nearest node lies within half
  a cell of the barrier move by <= 6.6e-4. Down barriers, discrete knock-in and every 1D layout are bit for bit, and
  so is every certificate anchor and golden (`docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/moved_values.txt`).
- Dated records without `initial_date` no longer accrue from the valuation date (coupons shrank as the contract
  aged).
- Phoenix `get_ko_payoff` keeps `accrued_offset`.
- The product, QUAD V2 and the intraday path agree on every Phoenix coupon period, the first one included.
- An aged MC REBASED knock-out accrues from the contract's origin (it lost the elapsed part).
- Ageing a KO-reset contract (`time_shift`, a theta bump) now moves an ABSOLUTE schedule 2 a day closer as
  well, and no longer grows its knock-out cash or knocked-in tenor. Schedule 2 used to stay where it was while
  the elapsed time was banked into `accrued_offset` (one day added 1/365 to every post-KI accrual, 0.75 ->
  0.75274), so KO-reset theta was wrong: about 10% low on a live contract, of the wrong sign on a knocked-in one
  (book contract: -0.0181 -> +0.0489). The one-day theta now equals the calculator's theta on the contract aged by
  hand (`test/accrual_time/test_ko_reset_theta.py`). A REBASED schedule 2 holds offsets and stays.

Measured on the PDE byte book (`test/pde_bitwise/capture_book.py`, 721 rows, main e3021668 vs this branch, same
machine): 678 rows bit-identical; in the 43 that moved only `bump_theta` moved, every price and every other Greek
is bit-identical. 16 Phoenix rows move at ulp level (relative 2e-14 to 2.5e-12: an aged numeric accrual now reads
the original quoted end instead of `(offset + bump) + (t - bump)`). 27 KO-reset rows move by the schedule-2 fix
above: relative 2e-2 to 1.6e-1 on the standard alive rows (e.g. `ko_reset|flat|standard` +0.02480 -> +0.02707),
up to a sign change on knocked-in and near-barrier rows (`3a|class|ko_reset|cont|knocked_in` -0.01807 ->
+0.04887, `3a|bumpcross|ko_reset|cont|85.5` -0.14185 -> +0.00369).

- Range accrual (equity `RangeAccrualOption`, FX `FxRangeAccrualOption` and its foreign and quanto variants): an
  annualized coupon accrues over the WHOLE contract -- `tenor`, else `initial_date` -> expiry in
  `range_config.day_count_convention` (which was never read), else a numeric contract's `maturity` plus the time
  `time_shift` has aged it by. It used to accrue over the REMAINING maturity while the in-range ratio counted every
  fixing, past and future, so the coupon shrank as the contract aged: a one-year 10% contract certain to pay 10.0
  priced at 7.34 after 91 days and 0.027 one day before expiry; it now prices at 10 x DF on every date. Inception
  prices are unchanged when the config's day count is the one the environment used.
- Range accrual theta: `time_shift` moves a numeric contract's fixing times with its maturity (they stayed put, so
  past fixings looked future), and a fixing the theta bump steps over is fixed at the held spot (a dated FX one used
  to be refused as an unrecorded past fixing). Theta on a certain coupon is now the discount roll (+8e-4 per day on
  the control) instead of a day's accrual lost (-0.026).
- cashleg `AccrualLeg`: `day_count` is now read. When the `LegSchedule` carries `period_start_dates` /
  `period_end_dates`, each period accrues `calculate_year_fraction(start, end, day_count)`; with times only it keeps
  `period_end - period_start` (bit-identical). The field used to be ignored and the dates never read, so ACT/365,
  ACT/360 and 30/360 US gave the same PV. On a dated quarterly 2026 leg (4% on 1,000,000, r = 5%, no KO): ACT/360
  38773.115547 -> 39306.910836 (+1.38%, = 365/360 x the dated ACT/365 value), ACT/365 -> 38768.460003 (90/91/92/92
  days instead of four quarter-years), 30/360 US unchanged. No library, test or example caller builds a dated
  schedule: every existing leg PV is bit-identical (14 caller configurations measured against dcea77b6). New
  refusals: period dates given for only one side or only some periods, non-`datetime` dates, an end not after its
  start; a dated `BUSINESS_DAYS` leg without the new `calendar` and `bus_days_in_year` fields (and those fields on
  any leg that would not read them); ACT/ACT ISMA/BOND/CHINABOND on a dated leg (no coupon frequency).

### Removed
- `AccrualConfig.accrual_dates` (it had no reader).

The analytic barrier engines' observation rules (the 3C handoff, spec
`docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md` §12). Prices inside the
barriers away from maturity are bit-identical.

### Fixed
- A spot beyond a discretely monitored barrier with no observation due is no longer a hit in
  `BarrierAnalyticalEngine`, `OneTouchAnalyticalEngine`, `DoubleBarrierOptionAnalyticalEngine` and
  `DoubleSharkfinOptionAnalyticalEngine` (and `SingleSharkfinOptionAnalyticalEngine`, which prices
  through the first two). The value is taken through the next observation
  (`engine/analytical/first_observation.py`): the Gaussian step to it and its barrier test are
  exact, the engine's own barrier-shifted formula applies from it on, and the expectation is a
  two-order Gauss-Legendre rule that raises `NumericalError` when the orders disagree beyond
  `Tolerance.RELATIVE`. Against the exact discrete recursion (`test/pde_routes/barrier_reference`),
  monthly and aged monthly schedules: single barrier 1.41 → 4.4e-2 (74 cells), one-touch /
  no-touch 0.74 → 2.2e-2 (18), double barrier 0.44 → 2.0e-2 (32), double sharkfin 0.58 → 1.3e-2
  (32), each below the shift's own error just inside the barrier (0.24, 0.096). An observation at
  the valuation instant still decides on the spot, at its own level. A schedule other than
  `STOP_FIRST_HIT` beyond its barrier now raises `PricingError` (it was priced as a hit). A price
  beyond the barrier costs 25-30 ms on the single barrier and up to 0.5-0.9 s on a daily-monitored
  double sharkfin (one Ikeda-Kunitomo series per quadrature node); inside, unchanged.
- `DoubleSharkfinOptionAnalyticalEngine`'s continuous paid-at-hit leg integrated the survival curve
  with 48 Gauss-Legendre nodes over (0, T), which miss its fall next to a barrier: off by up to
  2.4e-5 on a rebate of 5. The integral now runs on panels doubling from the time at which a
  barrier comes within nine deviations (exact below it), at two orders, and raises
  `NumericalError` when they disagree; 4e-15 against an adaptive reference on 60 cells.

### Changed
- A no-touch (`OneTouchOption` with `NO_TOUCH`, `DoubleOneTouchOption` with `DOUBLE_NO_TOUCH`) and
  both sharkfins refuse an `ACCUMULATE` observation schedule at construction (`ValidationError`):
  they pay only if the barrier is never hit up to and including expiry, so nothing accumulates
  (user ruling 2026-10-07). The PDE priced an accumulating no-touch as if only its maturity
  observation existed (4.304787 against 4.227537 under first hit, a year out, records at 0.5 and 1);
  the double-sharkfin MC summed every hit's cash. One-touches and barrier knock-outs keep the mode.
  The eight `3c|mat|*no_touch|acc|*` byte-book rows are retired.

### Added
- KO-reset snowball lifecycle tracking (spec `docs/superpowers/specs/2026-10-08-ko-reset-lifecycle-design.md`):
  `KOResetLifecycleTracker` books the schedule-1 knock-out, the knock-in that switches the contract to schedule 2
  for good (tested the same close, inclusive; schedule-2 records due on an earlier close are consumed), V0 at
  schedule 1's end on its tenor (no schedule-2 extension), and V1 at the knocked-in contract's end; a knock-out
  wins a tie. `PortfolioLifecycleManager` (equity backtest, dynamic scenarios), `ProductReplay` and the
  simulated-path backtest track KO-reset positions (the manager used to warn and reprice them as of every day,
  and every engine raised the day after schedule 1 ended un-knocked). Both schedule-2 modes: a knocked-in
  `REBASED` contract is priced as the `ABSOLUTE` contract it has become
  (`KnockOutResetSnowballOption.with_absolute_schedule_2`); an un-knocked `REBASED` contract needs a Monte
  Carlo engine (refused at configuration elsewhere, and in simulation surface mode). The knock-in close
  (`ki_date`) is the reset's timing; its accrual hangs from the persisted `KnockInAnchor` (the knock-in record that
  fired, at its contractual fixing; accrual time spec §5.1), stored in the state beside `ki_date`, carried by every
  pricing copy (`_otc_lifecycle_knock_in_anchor`) and embedded by
  `with_absolute_schedule_2(knock_in_time, *, anchor)` in each converted record's relative period and in the `post`
  terminal (§5.2): knock-in records due on days 91 and 93 observed on day 95 accrue from day 91. A carried `REBASED`
  knock-in needs `_otc_lifecycle_knock_in_date` and, when its knock-in records accrue, its anchor
  (`_otc_lifecycle_knock_in_anchor`); it is never looked up by date. Simulated paths take the lowest-index knock-in
  record due on a path's knock-in close and refuse records of different levels due on one close.

### Changed
- Every autocallable tracker holds its contract with numeric observation times snapped to the day it observes
  them (`round(time * 365) / 365`; whole-day times keep their exact float, dated records are untouched), and
  every consumer (pricing copy, replay, simulation surfaces) builds from it. An off-day numeric record used to
  be observed on its rounded day yet priced as still ahead (a day-1.4 knock-out: 4.9183 on the life surface
  against 0.4381). Snapping moves observation timing only: each record keeps its accrual period, so a `k / 12`
  Phoenix coupon at 12% still pays its quoted 1/12 (1.0) when observed on day 30 (the "cash follows the snapped
  time" amendment is withdrawn, accrual time spec §7). Whole-day and dated products are bit-identical; replay
  goldens unchanged.
- The replay and the simulation price the initial book before the first close (`product_for_pricing(...,
  initial=True)` keeps a record landing that day); the simulation keys such a state apart only where a record
  lands.
- Replay QUAD prices a KO-reset on `KOResetSnowballQuadEngine` (`SnowballQuadEngine` ignored schedule 2: 4.195
  for -1.744). A KO-reset under `vol_model != "bsm"` needs `vol_model_solver="mc"`: the vol-model PDE solvers
  price it on schedule 1 (21-26% off).

### Fixed
- KO-reset lifecycle final review: a `REBASED` schedule 2 hangs from the knock-in close, timed and dated from the
  run start, the clock pricing copies age on (a contract issued before the run placed it early, or its pricing
  copy held it the gap too far ahead) and sits exactly on the day grid (a record ~5.55e-17 years ahead on its own close made QUAD raise); a dated `REBASED` contract's
  knocked-in end is dated (it raised `TypeError` on the knock-in close); `observed_post_ko_indices` is PnL-explain
  bookkeeping; `PortfolioLifecycleManager` values each tracked autocallable from its tracker's initial copy and the
  dynamic scenario and equity backtest price their baseline after registering trackers (a carried knocked-in
  `REBASED` position was priced as booked); `EnsembleConfig` refuses a carried knock-in (it was dropped).
- `SnowballMCEngine`, `REBASED` KO-reset: a post-knock-in knock-out keeps the accrued offset of an aged contract
  without `initial_date` (an annualized control paid 6.5 instead of 7.5); V1 is read at each path's own maturity
  (`ki_time + max(offset)`, not the grid's last node) on that path's tenor (`get_maturity_payoff_v1` takes
  `contract_tenor`; an annualized `REBASED` V1 raised "ki_time is required").

### Removed
- `DoubleSharkfinOptionAnalyticalEngine(quad_points=...)`: the paid-at-hit integral controls its
  own error. Passing it raises `TypeError`.
- The double-sharkfin MC engine's `ACCUMULATE` aggregation branch (unreachable: the product refuses
  the mode).
- At maturity those engines observe only what is observed now: a discrete schedule's records at
  the valuation instant, each at its own level and paying its own cash (none: no hit; it tested
  the spot against the product's barrier, 5 for 35 and 25 for 7), and an untouched single-barrier
  knock-in pays its rebate (it returned 0).



Theta at an observation boundary (spec
`docs/superpowers/specs/2026-10-07-theta-observation-boundary-design.md`). Theta moves the contract
from the as-of date `t0` to the bumped date `t1` in one step shared by every caller
(`riskmeasures/greeks/theta_shift.py`): `GreeksCalculator` and the clock family (charm, color,
vega decay), the execution bump cells, `EquityPosition.get_trade_risk` with cash legs, and the
autocallable risk report with its surfaces. Lifecycle aging keeps its rules (`keep_landing=False`
runs the old comparisons) apart from the KO-reset fix below; the replay goldens are byte-identical.

### Changed
- An observation that lands on `t1` is kept at time 0 and observed at the new as-of instant; it
  was dropped. A daily accumulator one day before a fixing (spot 100, strike 95, 10,000 shares a
  day): theta +19,397 → +69,397, the difference being exactly that fixing's settlement.
- The observation at `t0` is decided on the spot, as the engines decide it at `t0`, and only its
  outcome is carried: a knock-in (on the product, as the autocallable engines read it, and now
  for the barrier family too), an accumulator fixing, a Phoenix's memory arrears, cash paid after
  `t1` as a receivable. A contract `t0` ends is worth its engine's own `t0` value re-discounted to
  `t1`, or 0 once paid within the bump: on an event day theta is about minus the payment; cash
  paid on `t1` itself stays. Theta excludes only cash flows already booked (user ruling 2026-10-07).
  An accumulator's `t0` fixing is that cash (it is no longer added to `past_observations`); a
  TERMINATION breach on it leaves the booked accrual and the rebate receivable; a KO-reset whose
  schedule 1 ends un-knocked matured then. A Phoenix `EXPIRY`-pay coupon decided at `t0` raises
  `CapabilityError`: it rolls up to the note's end, which no state carries yet.
- A carried knock-in is priced through each engine's own triggered knock-in, the path a knock-in it
  decides at the valuation instant takes, so theta differences one method: the PDE composition, the
  MC vanilla, the local-vol European kernel, QUAD's vanilla; the analytic engines' is the
  Black-Scholes vanilla; the Heston and SLV barrier engines (MC and 2D PDE) take it through a
  `knocked_in` flag on their kernels (every path knocked in; the PDE's own vanilla leg), equal to
  their own breached knock-in to 1e-12. A carried knock-in is an event that happened (user
  ruling 2026-10-08); they refused it.
- A schedule the bump empties is `ObservationSchedule.monitoring_ended` and priced as what
  remains (barrier family: vanillas and cash digitals, `engine/monitoring_ended.py`; a triggered
  knock-in the same way); theta there is carry and discounting, not 0. The Heston and SLV barrier
  engines raise `CapabilityError` on it (no remaining-European path under their own model).
- Charm, color and vega decay treat a contract `t0` ended as a fixed receivable (no spot or vol
  greeks at `t1`) and never difference a carried receivable.
- `time_shift(..., keep_landing=False)` on schedules, `BarrierConfig`, products and
  `AutocallableCashLeg`; product `time_shift` always ages and returns `None` (a KO-reset's returns whether
  every observation, schedule 2 included, has passed).
  `RangeAccrualOption.time_shift` now shifts numeric records and decides past fixings on the spot.

### Fixed
- `SnowballQuadEngine`, `PhoenixQuadEngine` and `KOResetSnowballQuadEngine` price an empty
  knock-in or knock-out side (the contract aged past its last date) as QUAD V2 does; they raised
  `PricingError`. A KO-reset whose schedule 1 ended un-knocked still raises: it has matured.
- `KnockOutResetSnowballOption.time_shift` ages an `ABSOLUTE` schedule 2 (the post-knock-in
  knock-out schedule); numeric post dates stayed put. A knocked-in KO-reset's theta now equals the
  knocked-in snowball's on schedule 2 (+0.0166 in the byte book's case, where it was -0.1406); 24
  KO-reset `bump_theta` byte-book rows move, no other output of the 713 rows. (The same fix landed
  independently on the accrual-time branch, entry above; this branch keeps both its `keep_landing`
  and accrual time's schedule-2 factor slicing. Both byte-book counts were measured against main on
  their own branch; the merged tree has not been re-measured.)
- `DiscreteQuadEngine` ignored a knock-in carried on the product (`_otc_lifecycle_knocked_in`) and
  priced a knocked-in contract as alive.
- The accumulator analytic and MC engines settle an observation at the valuation instant on the
  spot; the analytic engine raised on its zero-maturity leg.

## [0.10.0] - 2026-10-06

PDE 1D remaining routes (spec: `docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md`,
sub-project 3C). The four remaining 1D routes, the local-vol vanilla (`LocalVolPDESolver`), the
local-vol barrier (`LocalVolBarrierPDESolver`), the DCN (`DCNPDEEngine`) and the local-vol DCN
(`LocalVolDCNPDEEngine`), march on the shared PDE core: the shared presets, the cell-average
projection, grid Greeks, bump contexts, `explain` and grid-capable prepared sessions. Their
prices move toward independent references on every measured family. Before they moved, the
barrier family was repaired: the contract multiplier, the decision at the valuation instant and
at maturity, EXPIRY monitoring and the far-field row beyond a discrete barrier, a knock-in's
rebate and session grid, one bump layout per composite part; the analytic double-barrier and
double-sharkfin engines now read one double no-touch probability with error control. The Heston
and SLV barrier solvers' vol-of-vol → 0 limit moved onto the core as well. Last, under
discretely observed knock-in the not-knocked-in edge on the knock-in side takes the knocked-in
row, which moves 31 certified values in 10 cells, each as pre-measured. Every stage was checked
against the PDE byte book (510 rows at the base, 721 at the end) with its declared movers.

### Added
- `BarrierPDESolver` and `LocalVolBarrierPDESolver` accept `ObservationType.EXPIRY`: the product
  is a discretely monitored barrier whose only observation is maturity, its rebate paid at the
  terminal payment time (1.0e-5 from the closed form on the restated 0.7.0 refusal test).
- A knock-in's rebate is priced by `BarrierPDESolver`, `DoubleBarrierPDESolver` and
  `LocalVolBarrierPDESolver` (Fixed, below).
- Bump Greeks of a continuous knock-in through every `create_bump_context` consumer
  (`GreeksCalculator`, the bucketed Greeks, `execution/greeks.py`, portfolio, PnL explain); they
  raised `PricingError`. The bump context freezes one layout per composite part. On 12 cells
  against closed-form Greeks through the same bumps the largest errors go theta 4.6e-5 → 1.0e-6,
  rho 4.6e-4 → 1.8e-5, vega 9.5e-5 → 3.7e-5, delta 1.7e-4 → 6.9e-5.
- `LocalVolPDESolver` and `LocalVolBarrierPDESolver`: `explain`, `create_bump_context`, every
  `PDEParams` field honoured (the kernel's refusals of domain, event, damping and stepper fields
  and of a custom operator are gone), grid artefacts in prepared sessions;
  `LocalVolBarrierPDESolver.calculate_greeks` (`price, delta, gamma` from the grid, composed for
  a knock-in). The barrier route also gains per-observation barrier schedules and a lagged
  pay-at-hit rebate (the kernel raised `CapabilityError`), and both a value at maturity (the
  kernel raised `ValidationError` at `tau = 0`).
- `DCNPDEEngine` and `LocalVolDCNPDEEngine`: `calculate_greeks` (`price, delta, gamma` from the
  grid), `explain`, `create_bump_context`, every `PDEParams` field honoured, prepared sessions
  on the core adapters. Grid delta is within 0.040 % at six spots where the bump delta was up to
  0.85 % off; gamma one point above the knock-in barrier (0.113) is off by −0.0036 where it was
  off by −0.057.
- The DCN's session grid and spot curve, per level (the user's ruling on the plan's P8, in place
  of a fail-closed grid). `BranchSolution`, beside `CompositeSolution`, carries the alive and
  knocked-in surfaces. Each level is read from the surface the base-spot readout would take at
  that level, the knocked-in one at or below the knock-in barrier or when the contract is
  flagged knocked in at valuation, else the alive one, and its delta and gamma are
  differentiated within that surface. A session's grid rows equal `calculate_spot_greeks_curve`
  at the same levels bit for bit; a DCN flagged knocked in reads every row from the knocked-in
  surface. Nothing is refused.
- `localvol_greeks.local_vol_rho_theta`, the rho / theta half of `local_vol_model_greeks`, which
  the local-vol vanilla calls. The helper calls it too, so the Heston analytical, Heston MC,
  Heston-SLV MC and local-vol MC engines are bit-identical.
- A knock-in's session grid and one-solve spot curve, composed from its parts, on
  `BarrierPDESolver`, `DoubleBarrierPDESolver` and `LocalVolBarrierPDESolver`; `CompositeSolution`.
- `analytical/double_no_touch.py`: `double_no_touch`, the double no-touch probability by the
  method of images, written so that two large numbers are never added. It returns a value whose
  error bound is at most `Tolerance.ZERO` (1e-10) or raises `NumericalError` with the bound;
  `double_no_touch_with_bound` returns `(value, bound)` and refuses nothing. Over 10,543 cells no
  error is outside its bound and the largest is 0.46 of it. 1 to 13 image pairs, 4.5 to 35 µs a
  call against 45 to 49 µs for the 80-term series it replaces. The double-barrier and
  double-sharkfin engines read it.
- Core adapters `LocalVolEuropeanAdapter`, `LocalVolBarrierAdapter`, `DCNAdapter`,
  `LocalVolDCNAdapter`; solver mixin `LocalVolSurfaceMixin`, one copy of the surface members of
  the five local-vol solvers (bitwise for the local-vol Snowball and Phoenix).
- `BaseEngine.static_replication(product, env)`: the legs an engine prices a product with when
  it is a static combination of other products (`None` otherwise); `GreeksCalculator`
  computes such a product's greeks as the weighted sum of its own greeks of the legs.
- `quantark.asset.equity.engine.pde.core.adapters.expiry_replication`: EXPIRY monitoring on
  the double-barrier and touch PDE solvers as European vanillas and cash digitals on the
  analytic engines (`FamilyAdapter.replication`; no grid).
- `quantark.volmodels.heston.deterministic_limit.DeterministicVarianceLocalVol`: the
  vol-of-vol → 0 limit of Heston and Heston-SLV as a local-vol surface, `L(S, t) sqrt(v(t))` with
  `v(t) = theta + (v0 − theta) e^(−kappa t)` (`L = 1` under Heston).
- `quantark.modelvalidation amend` keeps an imported certificate's EXTERNAL benchmark: it never
  runs it, re-runs every candidate, carries a bit-identical cell verbatim, translates a moved one
  by the harness's own interval arithmetic, and refuses any translation that would change a
  verdict (`RELEASE_PROCEDURE` section 10). It refused the whole study before.
- `quantark.modelvalidation amend --rerun-candidate NAME` (repeatable) re-runs a candidate whose
  implementation changed although its schema-1 identity (its configuration) did not; the
  amendment records `rerun_candidates` and an unknown name is refused (`RELEASE_PROCEDURE`
  section 6).
- `test/pde_routes/`: the exact DCN and discrete-barrier recursions with their term-curve forms,
  the no-touch series, the local-vol barrier kernel as the test oracle (`lv_barrier_kernel.py`),
  and their guards.

### Changed
- **The four migrated routes' prices and Greeks**, toward the references on every measured
  family. Local-vol vanilla and barrier, maximum |error| at `standard` in price units at spot
  100, kernel → core, on a constant / the byte book's / a steep surface: vanilla 4.6e-3 / 4.8e-3
  / 4.8e-3 → 2.5e-4 / 2.6e-4 / 2.9e-4; continuous barrier 3.5e-3 / 3.8e-3 / 3.2e-3 → 4.1e-4 /
  4.0e-4 / 7.1e-4; monthly 5.9e-3 / 4.7e-3 / 2.2e-2 → 2.5e-4 / 2.5e-4 / 6.2e-4; weekly 1.2e-2 →
  1.6e-4 (constant); daily 1.1e-2 / 8.3e-3 / 9.4e-2 → 1.2e-3 / 1.1e-3 / 2.1e-3. Greeks on the
  constant surface, delta and gamma: vanilla 5.7e-4, 1.3e-3 → 4.7e-6, 1.8e-6; continuous barrier
  7.8e-4, 1.3e-3 → 1.1e-5, 1.3e-6; discrete barrier 1.7e-3, 9.1e-4 → 8.9e-5, 4.4e-5; vanilla rho
  and theta 1.5e-4, 2.6e-5 → 1.1e-5, 1.2e-6. DCN, in bp of notional over 24 cells (four
  contracts, six spots), maximum at fast / standard / high 5.2 / 7.2 / 7.7 → 0.38 / 0.35 / 0.10
  (median 2.1 / 1.4 / 1.8 → 0.12 / 0.11 / 0.03); the points ladder at the money no longer
  wanders (−10.5 to +7.7 bp over 201 to 2,401 points → 0.10 to 0.46 bp); the local-vol DCN on a
  skewed surface −0.9 to −9.8 bp → within 0.18 bp of its finest rung. On kinked rate and dividend
  curves: vanilla 4.5e-3 → 2.6e-4, monthly barriers 5.1e-3 → 4.6e-4, DCN 4.2 to 6.0 bp → 0.22 bp.
- **Their defaults are the shared presets**: 200 / 400 / 800 concentrated points, 4 / 4 / 8 steps
  per day, half-step damping (1, 1), cell-average projection. `GridConfig(time_steps=200)`
  reproduces the old local-vol time budget at a cost of at most 1.4e-5 on a vanilla and 3.4e-5 on
  a continuous barrier, but up to 5.4e-4 monthly, 4.3e-3 weekly and 2.6e-2 daily. Daily
  monitoring at `standard` carries about 1e-3 of time error and wants 8 steps per day (+9.3e-4 /
  +2.1e-4 / +1.1e-5 at 4 / 8 / 16). The DCN's 201-point floor is gone (`fast` is 200 points), and
  a DCN grid of 2,401 points needs `GridConfig(max_points=2401)`. Their wall time: Performance,
  below.
- `LocalVolPDESolver.calculate_greeks`: `delta` and `gamma` are pointwise grid Greeks, not 1 %
  bumps (the convention moves them by at most 2.4e-4 and 2.3e-5 on the measured vanillas); `rho`
  and `theta` keep their definitions and units and re-solve on the bump context with the surface
  held fixed. At maturity it returns `price, delta, gamma`. A numerical vega through
  `GreeksCalculator` re-prices on that context's frozen layout; a fresh grid under the shifted
  surface differs from it by the layout's own movement (6.2e-6, 1.6e-5 relative). The
  structured vol-model risk's local-vol re-price is the solver's price with the surface held
  fixed, and so now includes a lagged product's settlement discount.
- `DCNPDEResult.num_time_steps` is the delivered step count (2,953 at `standard` on contract A,
  where it read 485), `num_space_nodes` the layout's; the DCN's horizon is its final observation
  date.
- A single- or double-barrier knock-in with a rebate (Fixed). A knock-in whose schedule records
  carry a payoff different from its rebate, or whose schedule aggregates other than
  `STOP_FIRST_HIT`, raises `ValidationError`; four 0.7.0 nodal-semantics goldens in
  `test_pde_base_family_projection` (knock-ins with per-record payoffs, two under `ACCUMULATE`)
  now assert that error.
- A discretely monitored single barrier's far-field row on the barrier's side is the certain
  knock-out at the next observation when the edge is beyond that observation's level; with no
  observation ahead, an edge inside the level, on the opposite side and under `ACCUMULATE` it is
  the vanilla asymptote, as before. EXPIRY cells move by −2.8e-4 / −1.2e-4 (constant surface);
  monthly and daily cells do not move to the printed digits; a two-year barrier observed only in
  its second year moves by 1.0e-5 and 1.5e-5. Under `bounds=(None, 140)` on a monthly up-and-out
  call the old row was off by +1.70 / +9.18, the new one by −8.8e-3 / −1.6e-2.
- The European, barrier and double-barrier PDE solvers price per contract (Fixed).
- A discretely monitored barrier, double barrier or touch product valued with the spot beyond a
  barrier and no observation due is solved, not short-circuited (Fixed).
- `calculate_spot_greeks_curve` and the session `GRID` output of a barrier-family product: a
  knock-in's rows are the composition of its parts, at the vanilla part's nodes by default; a
  knock-out's or touch product's rows at hit levels are the hit outcome with zero delta and
  gamma; a product decided at the base spot raises `CapabilityError` for a session grid and
  returns `reprice` rows from the public curve (Fixed).
- `DoubleBarrierOptionAnalyticalEngine`: every price with a rebate under continuous or discrete
  monitoring, and every EXPIRY price with the spot outside the corridor (Fixed). Rebate-free
  prices and EXPIRY inside the corridor are bit-identical.
- `DoubleSharkfinOptionAnalyticalEngine`: the continuous paid-at-hit rebate leg (up to 4.2e-6 at
  vol 22 % and 3.2e-5 at vol 8 % on a rebate of 5); every price where its 80-term series was
  unresolved; `quad_points` is keyword-only. Every other leg moves at round-off (at most 3.6e-15
  at equity parameters).
- A barrier-family product with a record on the valuation date: its decision there carries that
  record's level, cash and payment time and is applied at the spot, not written into the grid.
  At maturity `price`, `calculate_greeks` and the session outputs return one terminal value,
  which carries the participation and the terminal payment discount `DF(T_pay) / DF(T)`, and its
  delta the same factors; a touch product's `calculate_greeks` there agrees with its `price`; an
  expired knock-in prices. `EuropeanPDESolver`'s session outputs and `AmericanPDESolver`'s
  `calculate_greeks` and session outputs at expiry carry their `price`'s payment discount
  (Fixed).
- `DoubleBarrierOptionAnalyticalEngine` and `DoubleSharkfinOptionAnalyticalEngine` raise
  `NumericalError` on an input whose no-touch probability is not resolved to `Tolerance.ZERO`.
  On the double barrier that takes a vol below 1e-4 and a forward carried onto a barrier; inside
  the market objects' limits nothing with a vol of 1e-4 or more is refused, and the sharkfin's
  floor is 1e-3.
- A prepared session asked for more than `PV` with a lifecycle state: the facade's values on an
  engine that supports a state, `CapabilityError` on one that does not (Fixed).
- **The knock-in-side far-field row** under discretely observed knock-in, while a knock-in
  observation remains: every European-knock-in Snowball, Phoenix and KO-reset price, flat,
  local-vol and 2D (Fixed). Under monthly or daily knock-in it moves at round-off.
- Nodal coefficient sets never collapse to the scalar stencil: a one-step local-vol solve no
  longer returns the flat-vol price (5.676243 exactly). Bitwise on every byte-book row.
- Sessions and the registry: `EquityLVPDESessionAdapter` is grid-capable (the artefact-chain
  `prepare` moved up from the local-vol autocallable adapter, bitwise for the local-vol Snowball
  and Phoenix). `LocalVolPDESolver`, `LocalVolBarrierPDESolver` and `LocalVolDCNPDEEngine`
  register on it, `DCNPDEEngine` on `EquityPDE1DSessionAdapter`. A prepared price equals the
  direct one bit for bit for the four, and a `PV + GRID` session's rows equal the public curve.
  `execution/inventory.py` and `docs/execution/capability-matrix.md` list the four as
  prepared-capable.
- `ROUTE_RULES` keeps the 2D row only. Its refusal now ends "These are the 2D route's limits; no
  spec is committed to moving this route onto the shared core".
- Goldens: nine replay frames moved, all one Snowball with discretely observed knock-in in the
  scalar-BSM and local-vol books on its one not-knocked-in day (2024-01-02): price −3.18e-5 /
  −3.28e-5, and every later row's P&L by the same offset. Every `book_*` frame is byte-identical
  and no execution golden moved (`test/replay_golden/README.md`, the 2026-10-06 section).
- Examples, re-run against the 0.9.0 tree: `dcn_gate_report_demo` DCN-A PDE −93,294.8 →
  −93,291.5 and DCN-B −150,768.0 → −150,579.8 (|dPV| against Monte Carlo 115.6 → 112.3 and
  277.0 → 88.7); `vol_models_demo` local-vol PDE 8.8317 → 8.8275, Dupire market-IV vega 38.6555
  → 38.7113; `bucketed_greeks_consolidated_api_demo` Dupire market-IV parallel 0.32127 → 0.32395
  and `strike_column[2]` 0.16597 → 0.20443; `mo_volmodels/07_barrier_exotic` local-vol PDE 23.402
  → 23.338 (gap to Monte Carlo 1.7 % → 1.4 %); `mo_volmodels/09_delta_hedging` local-vol delta0
  0.3736 → 0.3733, gamma0 8.1e-5 → 2.7e-4. `mo_volmodels/03_dupire_localvol`'s overall LV RMSE
  goes 4.519 → 4.872 vol-points, which is not a regression: against the converged price of the
  same Dupire surface the core's 300 × 150 solve errs 0.017 rms (0.043 max) and the kernel's
  0.57 rms (2.05 max), so the kernel's discretization error partly cancelled Dupire's fit error.
  `mo_volmodels/11c_grid_preflight.py` binds on the solver's `grid_binder`.

### Removed
| Removed | Replacement |
|---|---|
| `GridLayerMixin` (and its export from `pde.grid`) | `BasePDESolver` |
| `LV_UNIFORM_ACCURACY_PROFILES`, `DCN_ACCURACY_PROFILES`, `resolve_config(profiles=)`, `GridBinder(profiles=)` | `ACCURACY_PROFILES` |
| `DAMPING_DEFAULTS["local-vol"]`, `DAMPING_DEFAULTS["DCN"]` | `DAMPING_DEFAULTS["core"]` |
| the `"local-vol vanilla"`, `"local-vol barrier"` and `"DCN"` rows of `ROUTE_RULES` | none: the fields are honoured |
| `DCNPDEEngine.price_detailed(layout=)`; the constructor's `points >= 201` check | `PDEParams(grid=…)`; the shared presets |
| `dcn_pde_solver._sinh_grid`, `_operator_coeffs`, `_theta_step`; `LocalVolDCNPDEEngine._step_vol` and its `1e-8` floor | the core march; the mixin's midpoint local-vol rows |
| `price_barrier_lv_pde`, `_barrier_boundary_value`, `_align_discrete_barrier_midcell`, `_solve_ko_lv` (the barrier half of `quantark.volmodels.localvol.pde_kernel`) | `LocalVolBarrierPDESolver`; the functions live on in `test/pde_routes/lv_barrier_kernel.py` as the test oracle |
| `quantark.volmodels.heston.pde_kernel._deterministic_barrier_price`, `quantark.volmodels.slv.slv_pde_kernel._deterministic_slv_barrier` | the engines' deterministic limit on the core (`DeterministicVarianceLocalVol` on `LocalVolBarrierPDESolver`) |
| `DCNLocalVolPDEAdapter`, `_dcn_pde_adapter` | `EquityLVPDESessionAdapter` |
| the `PricingError` on EXPIRY monitoring in `BarrierPDESolver` | EXPIRY monitoring (Added) |
| `DoubleBarrierOptionAnalyticalEngine._prob_inside_at_expiry` | `analytical/double_no_touch.py` |
| `DoubleSharkfinOptionAnalyticalEngine(max_terms=…)`, its `DEFAULT_MAX_TERMS`, `_survival_probability_continuous`, `_weighted_sine_integral` | `analytical/double_no_touch.py`; there is no term count to set |

### Fixed
- **The contract multiplier on the European, barrier and double-barrier PDE solvers.** Their
  value at `contract_multiplier` 10 was their value at 1 (ratio 1 in price, delta and gamma),
  where the analytic engines, the kernel local-vol routes and `AmericanPDESolver` scale by 10. An
  expired product did scale, so the route disagreed with itself, and a knock-in's part products
  were built without the multiplier. The payoff, event and rebate cash, the Dirichlet rows, the
  hit outcomes and the knock-in's parts now carry it, `(participation × option + rebate) ×
  multiplier`. Multiplier 1 is bit-identical.
- **The valuation-date hit test under discrete monitoring.** A spot beyond a discretely
  monitored barrier with no observation due returned the rebate or the vanilla; it is now solved
  from its spot. Error against the exact discrete recursion: monthly up-and-out call, rebate 5,
  spot 135 / 131, −0.202 / −0.346 → +5.3e-5 / +8.3e-5; observed only in months 7 to 12, −1.08 →
  +1.2e-4; monthly down-and-out put 85, rebate 3, spot 80, +3.2e-2 → −2.0e-5; monthly up-and-in
  call at 135 / down-and-in put at 80, +0.596 / +9.8e-2 → −1.7e-5 / −1.9e-5; monthly double
  knock-out 80 / 130, rebate 5, spot 135, −8.6e-2 → +5.8e-6; monthly one-touch paid at expiry,
  spot 135, +7.9e-2 → −3.5e-6; an `ACCUMULATE` schedule paying 5 monthly at spot 135 (closed
  form 72.55), −67.7 → −1.1e-4. The touch solvers move by 0.08 (monthly) to 3.7 (semiannual). The
  double barrier's `calculate_greeks` at a continuous hit took the rebate undiscounted (5.000000
  for 4.962640 under a quarter-year lag at 3 %); it now takes `price`'s value.
- **The knock-in rebate.** The knock-in composite built its knock-out leg with rebate 0 and
  dropped the rebate. On a rebate of 5 the single barrier was off by −3.747 (continuous up-and-in:
  the rebate-free price), −2.588 (continuous down-and-in) and −4.852 (monthly: the whole `R ×
  DF`), the double barrier (corridor 80 / 130) by −2.2205 / −2.2206 (call / put). A knock-in is now
  `participation × vanilla + R × DF(0, T_pay) − leg(R)`: +9e-5 to +4.9e-4 on 8 single-barrier
  rebate cells, the rebate-free cells' own error, and +2.5e-4 / +1.2e-4 on the double barrier. A
  knocked-in product forfeits the rebate; `R = 0` is bit-identical.
- **A knock-in's session grid and spot curve, and a knock-out's rows at hit levels.** A knock-in
  was marched as it stands under `SESSION_OUTPUTS`: on a continuous up-and-in call 130 the public
  curve was off by −4.19 at the money and −51.7 at 150, and a session's `PV + GRID` returned
  2.85597 for 6.93002. The rows are now the composition of the parts, within 5.2e-4 in price,
  6.6e-5 in delta and 7.5e-5 in gamma of the closed forms and the exact recursion. A knock-out's
  or touch product's row at a hit level carried the edge node's value and its one-sided
  gradients; it is now the hit outcome with delta and gamma 0 (from −0.150 on an up-and-out call
  with no rebate, −0.011 with a rebate of 5, +0.028 on a one-touch), and the double knock-out's
  price there goes 4.85223 → 5.00000, the facade's value. A product decided at the base spot
  returned a grid although the facade's hit test never ran; it now raises `CapabilityError` for a
  session grid, and the public curve returns `reprice` rows.
- **The analytic double-barrier engine's rebate terms and its EXPIRY branch outside the
  corridor.** The knock-out added `R × DF` on every path and the knock-in used the probability of
  ending inside the corridor. Corridor 80 / 130, strike 100, one year, rebate 5: continuous
  knock-out +2.221 / +1.533 / +1.125 (spots 100 / 90 / 120) → at most 4e-15, knock-in +1.309 /
  +1.629 / +1.859 → at most 5e-15; monthly knock-out +1.80 to +2.95 → +0.017 to +0.093 and
  knock-in +0.90 to +1.53 → −0.017 to −0.093, the barrier shift's own error. Under EXPIRY a spot
  outside the corridor returned the rebate or the vanilla, −3.350 to +3.497 on eight cells; the
  closed form now applies from any spot, at most 1.1e-14.
- **A record on the valuation date.** The march wrote the valuation-date event into the nodes,
  and the facade tested the product's own barrier. On an up-and-out (barrier 130, rebate 5) with a
  record at 120 paying 7 (worth 6.947696) the spots from 120 outward returned 5.00 for 6.95, and
  just inside the record's level the price was off by −3.7e-2 / −5.2e-2 (delta −0.27). The
  decision now carries the record's level, cash, payment time and operation and is folded at the
  spot onto the continuation: exact where it fires, +5e-6 / +1.2e-4 just inside. A value read
  next to a record crossed the jump, up to +2.52 in price and +13.3 in delta on an up-and-in call
  130 at 129.9 to 129.99; now within 2.0e-4 and 8e-6. On 168 cells (single and double knock-out,
  both one-touches and both no-touches, lags 0 and 0.25, flat and kinked curves) a fired record
  is its cash to the last bit and the rest is within 2.8e-4 in price and 1.0e-5 in delta of the
  term recursion, where the old values were off by −1.2 to −3.8 at the record's lower level, +1.4
  to +4.5 just inside its upper level and −1.7 to −2.0 beyond the product's barrier. Under
  `ACCUMULATE` at and beyond the level: −37.6 to −43.0 → within 1.4e-4.
- **The barrier family at maturity.** At `tau = 0` the expired path returned `product.get_payoff`:
  no payment discount, no participation, and for a touch product a number that ignored the touch;
  `calculate_greeks` disagreed with `price`, and an expired knock-in raised. A value is now the
  terminal payoff × participation × `DF(T_pay) / DF(T)`, then the decision, under every
  operation. A knocked-out product's `calculate_greeks` returns the cash, not the payoff (35 →
  5); an expired knock-in prices and pays its rebate when untriggered (`calculate_greeks` 25 →
  5) and takes a bump context; a maturity record's own level and cash apply (25 → 7); a schedule with no maturity record
  does not test the barrier there (5 → 35). With the terminal payment a quarter-year after
  maturity at 3 %, a surviving knock-out, an untouched no-touch and a triggered or untriggered
  knock-in take the discount (25.000000 → 24.813201, delta 1 → 0.992528), and the value no longer
  jumps by 0.187 at the last instant. Participation 0.5 applies (25.000000 → 12.406601). A touch
  product's `calculate_greeks` there agrees with its `price`: a touched one-touch reports its cash
  (0 → 5), a touched no-touch nothing (5 → 0). `AmericanPDESolver.calculate_greeks` and session
  outputs at expiry under the same lag (25.000000 → 24.813201, delta 1 → 0.992528) and
  `EuropeanPDESolver`'s session outputs (25.000000 → 24.813201) take their `price`'s discount. An
  expired product with no lag, participation 1 and multiplier 1 is bit-identical.
- **The lifecycle state of a `PV + GRID` session on a European.** The session passed the state to
  `price` on the `PV` path and to nothing on the other. A European call with one realized, unpaid
  cashflow of 7 due in 91 days returned the live 9.601810 under `PV + GRID` whether terminated
  (price 6.947839) or alive with the receivable (16.549649). Now a terminated claim's grid fails
  closed (`CapabilityError`), and a live one returns 16.549649 with 6.95 in each row. The barrier
  and Snowball engines, which do not support a state, returned 2.671787 and 97.157139; they now
  raise `CapabilityError`, as for `PV`. A request with no state is bit-identical.
- **The double sharkfin's no-touch probability.** The engine's 80-term eigenfunction series had
  no convergence control: corridor 80 / 130, spot 129.9, vol 1 %, one day returned 0.248354 for
  0.858487, putting a rebate of 5 off by 3.050664, and over 637 cells its worst error was 1.0 at
  dimensionless times up to 1e-3 (two days at vol 20 % on that corridor). It now reads
  `double_no_touch` at its three call sites. Its continuous paid-at-hit factor, which was off by
  9e-8 to 6.3e-6, is within 1.3e-9 at spots 85 and 100 and 6.2e-8 at 127 (one year, vol 22 %).
  The price moves by up to 4.2e-6 (vol 22 %) and 3.2e-5 (vol 8 %) on a rebate of 5, and by up to
  a whole rebate where the series was unresolved.
- **The Heston and SLV barrier solvers' vol-of-vol → 0 limit.** Below `_SIGMA_MIN` (1e-4)
  `HestonBarrierPDESolver` and `HestonSLVBarrierPDESolver` used the 1D local-vol kernel at a
  constant effective vol `sqrt(average variance)` (for SLV, with a 15 × 5 tabulation of the
  leverage). That is exact only for Europeans; for a barrier it was a model error. They now march
  the exact deterministic-variance local vol `L(S, t) sqrt(v(t))`, `v(t) = theta + (v0 − theta)
  e^(−kappa t)` (`DeterministicVarianceLocalVol`), on `LocalVolBarrierPDESolver` at its standard
  preset, after the 2D solvers' own refusals. Measured at the standard preset against a
  3,201-point, 40 steps-per-day reference: with flat variance against the Black–Scholes barrier
  closed form, core 1.84e-4 (continuous up-and-out call 130) / 3.17e-4 (down-and-in put 80),
  old 3.06e-4 / 4.58e-3; monthly up-and-out 125, core 1.27e-4, old 2.08e-3; with decaying
  variance (v0 0.09, theta 0.04, kappa 1.5) the old route was 7.0e-3 to 3.25e-2 from the exact
  limit, the core within 2.0e-4; SLV with a skew leverage, old 1.42e-2 to 5.23e-2, core within
  1.9e-4. Heston and SLV barrier prices at a vol-of-vol below 1e-4 move by these amounts. The
  grid spans the model's vol, the larger of the environment's strike vol and the largest
  `sqrt(v(t))` to maturity (`DeterministicVarianceLocalVol.domain_vol`), so an environment
  quoting less than the model does not narrow it. The
  low-level `price_barrier_heston_pde` / `price_barrier_slv_pde` now raise `ValidationError`
  below `_SIGMA_MIN`, naming the engine.
- **The knock-in-side far-field row.** The not-knocked-in lower row of a standard product was
  "principal plus rebate to maturity", as if the far edge never knocked in; with one knock-in
  date at maturity nothing resets it. Under discretely observed knock-in, European included, that
  row now takes the knocked-in row (the upper row for a reverse product) while a knock-in
  observation remains at or after the row's time; after the last one, on an exhausted schedule,
  or for a KO-reset after its first schedule, it is the old row. Error against the exact recursion
  (QUAD for the reverse product): Snowball with European knock-in, spot 100, standard / high,
  +4.37e-3 / +4.25e-3 → +4.2e-5 / −4.9e-5; spot 86, +4.73e-3 / +4.75e-3 → −1.6e-4 / −8.9e-5; vol
  35 %, +5.03e-3 / +5.10e-3 → −2.0e-4 / −4.4e-5; Phoenix with knock-in at maturity only, spot 100
  / 80, +3.3e-3 / +4.0e-3 → −1.7e-4 / −1.2e-4; reverse Snowball, spots 100 / 114 / 98, +2.96e-3 /
  +3.88e-3 / +1.98e-3 → −1.8e-4 / −1.2e-4 / −2.3e-5; Snowball under `bounds=(60, None)` / `(70,
  None)`, +1.255 / +4.295 → −1.2e-3 / −0.129. On the byte book's flat market against QUAD at
  32,001 points: Snowball price +4.17e-3 → +3.0e-5 (exact recursion +4.17e-3 → +2.8e-5), Phoenix
  with knock-in at maturity +3.03e-3 → −2.2e-4, reverse Snowball +7.61e-3 → −5.3e-4; bump vega
  +5.18e-3 → −2.2e-5, +4.08e-3 → −7.5e-6, +8.90e-3 → −1.2e-5. The reconciliation identity stays
  at round-off, the knock-in probabilities do not move, and the maturity leg's error falls from
  +4.70e-3 to +3.7e-4.
- **EXPIRY monitoring on the double-barrier and touch PDE solvers.** `DoubleBarrierPDESolver`,
  `OneTouchPDESolver` and `DoubleOneTouchPDESolver` tested the barrier during the march under
  `ObservationType.EXPIRY`, close to continuous monitoring: at spot 100, vol 20 %, r 3 %, q 1 %,
  one year, 0.9.0 priced a 80/130 double knock-out call at 3.144 (exact 4.848), an up one-touch
  at 120 paying 10 at 3.428 (exact 1.756) and a 85/120 double one-touch at 7.237 (exact 3.777).
  A barrier observed only at maturity makes the value a function of the terminal spot alone, so
  before maturity these three solvers now price it by static replication: European vanillas and
  cash-or-nothing digitals on the product's own expiry and settlement, priced on
  `BlackScholesEngine` and `DigitalOptionAnalyticalEngine` (a knock-out is the vanilla inside
  the corridor plus the rebate outside it; a knock-in the vanilla outside plus the rebate inside;
  a touch product its digitals). Against quadrature of the terminal value the price is within
  1e-9 relative, with and without a payment lag, and knock-out plus knock-in equals the vanilla
  plus the rebate on a flat and a smile market. Each leg reads its own strike's vol, so under a
  smile the value is the smile's static replication, not the march's single strike-selected vol.
  No PDE grid is requested or bound for such a product: bump contexts return the engine itself,
  a session prepares no artefact, and a session `GRID` must name its levels
  (`grid_spot_levels`; without them `CapabilityError`, as there are no nodes). Greeks follow
  `GreeksCalculator`: the new hook `BaseEngine.static_replication` exposes the legs, and
  `GreeksCalculator.calculate` returns the weighted sum of its own greeks of each leg under the
  caller's method, request and bump configuration (closed forms for the vanillas under `auto` /
  `analytical`, bumps for the digitals, its own refusals per leg). The solver's
  `calculate_greeks` and the session rows (`calculation_mode` `"static_replication"`) are
  `GreeksCalculator(params=solver.params)` on the legs: delta within 1.27e-3 and gamma within
  3.6e-5 of the exact derivatives (relative, 80 cells). The single barrier keeps its EXPIRY
  far-field row; at maturity the terminal value applies as for every barrier product.
- **`GreeksCalculator`'s closed forms under a terminal payment lag.** The analytic route
  (`method="auto"` / `"analytical"` on a European vanilla) ignored `settlement_convention`: at a
  quarter-year lag an at-the-money call returned price 96.0158, delta 5.7385, rho 4.7783 where
  `BlackScholesEngine.price` and the bumped route give 95.2983, 5.6950, 4.5049. The closed forms
  now take the lag as the engine prices it, `V = DF(T, T_pay) V_T`: price, delta, gamma, vega,
  dividend rho, theta (and its three-way split) and the extended closed forms scale by
  `DF(T, T_pay)`, and rho adds the payment's own rate sensitivity, `-(T_pay - T) V_T` (4.5044).
  At expiry the payoff is discounted over the lag, rho is `-lag V` and theta its accretion. A
  vanilla paid at expiry is bit-identical.

### Breaking changes
- `GridLayerMixin` and its export from `pde.grid` are removed; a solver subclasses
  `BasePDESolver`.
- `LV_UNIFORM_ACCURACY_PROFILES` and `DCN_ACCURACY_PROFILES` are removed, and the `profiles=`
  keyword is removed from `resolve_config` and from `GridBinder`: every 1D route takes the shared
  `ACCURACY_PROFILES`.
- `DAMPING_DEFAULTS["local-vol"]` and `DAMPING_DEFAULTS["DCN"]` are removed; the routes use
  `DAMPING_DEFAULTS["core"]`.
- `DCNPDEEngine.price_detailed` no longer takes `layout=` (passing it raises `ValidationError`
  naming `PDEParams(grid=GridConfig(...))`, from `REMOVED_DCN_NAMES`), and the constructor's
  `points >= 201` check is gone. `DCNPDEResult.num_time_steps` is the delivered step count. A 2,401-point DCN
  grid needs `GridConfig(max_points=2401)`.
- `LocalVolPDESolver.calculate_greeks`: `delta` and `gamma` are pointwise, not 1 % bumps.
- The barrier half of the 1D local-vol kernel (`price_barrier_lv_pde` and its helpers) leaves
  the library for `test/pde_routes/lv_barrier_kernel.py`.
- New `ValidationError`s: a knock-in whose schedule records carry a payoff different from its
  rebate, or whose schedule aggregates other than `STOP_FIRST_HIT`; `price_barrier_heston_pde` /
  `price_barrier_slv_pde` below `_SIGMA_MIN` (the 2D barrier kernels refuse the vol-of-vol → 0
  limit and name the engine that prices it).
- New `CapabilityError`s: a session grid for a barrier-family product decided at the base spot;
  a stateful rich-output session request (`PV + GRID`, event statistics) to an engine without
  lifecycle support; a session grid for a terminated European.
- `DoubleSharkfinOptionAnalyticalEngine`: `max_terms` is removed and `quad_points` is
  keyword-only, so a second positional argument written for `max_terms` raises.
- New `NumericalError` from `double_no_touch`, and so from `DoubleBarrierOptionAnalyticalEngine`
  and `DoubleSharkfinOptionAnalyticalEngine`, on an input whose no-touch probability is not
  resolved to `Tolerance.ZERO`.

### Per-stage deltas (PDE byte book)
Stage 1, the barrier family:
- (h) analytic engines: 510 rows, 0 diffs (no PDE file changed).
- (e) multiplier: 510 rows bit-identical (every row has multiplier 1); 5 added (`3c|mult10|*`).
- (f) valuation instant: the record-level event builders bit-identical (515 rows); then exactly
  the eight `3a|farfield|{di_put_85,do_call_85}|*|80|*` rows move, equal to the evidence's
  `r5_book_hit_rule.json` bit for bit, up to 0.5762 in price, 0.1878 in grid delta and 4.306e-2
  in grid gamma (`3a|farfield|do_call_85|kinked|80|lag0`); 21 added.
- (f) maturity: 536 rows bit-identical, both expired barrier rows included; 102 added.
- (b) EXPIRY and the far-field row: discrete-barrier rows at round-off (bump rho 5.3e-13, bump
  vega 4.1e-13); `3c|vdec|uo_late|beyond` 3.39e-5 in price, toward the exact recursion (+6.1e-5
  → +2.7e-5 off), and `3c|vrec|uo|119.99` (+8.6e-5 → +8.0e-5); the `3a|farfield` rows
  bit-identical; 5 added.
- (g) session grid: 643 rows bit-identical; 41 added.
- (g) lifecycle on rich outputs: 684 rows bit-identical; 3 added.
- (a) knock-in rebate: 687 rows bit-identical; 16 added.
- (c) per-part bump layouts: eight `knock_in|*|bump` rows go from `PricingError` to values;
  `knock_in|down_in_put|knocked_in|bump` within 6.1e-5; `3a|base6|di_put|{default,v070}` (a
  monthly down-and-in put) in bump Greeks only, gamma 1.82e-4, delta 1.63e-5, vega 4.24e-6, rho
  9.80e-7, theta 1.07e-7 (the evidence's both-frozen arm); 1 added.
- (d) nodal sets: 704 rows, 0 diffs.

Stage 2, the routes:
- (0) `LocalVolSurfaceMixin`: 704 rows, 0 diffs.
- (a) local-vol vanilla: `lv_vanilla|{default,600x200}`, price up to 2.31e-3, grid gamma 2.08e-3,
  grid delta 8.67e-5; against a fine-grid core ladder the price error goes 2.57e-3 → 2.60e-4 /
  1.14e-3 → 2.79e-4 and delta 8.65e-5 → 2.5e-7 / 5.2e-5 → 5.8e-7; 3 added.
- (b) local-vol barrier: six `lv_barrier_*` rows, price up to 2.01e-3
  (`lv_barrier_discrete|default`). Against the oracle's Richardson references five move toward
  (discrete default 1.75e-3 → 2.5e-4) and `lv_barrier_cont|600x200` away by 1.8e-5 (6.2e-5 →
  8.0e-5). The local-vol autocallables are bit-identical; 2 added.
- (c) DCN: `dcn|default` and `dcn|601`, pv up to 336.1 (`dcn|601`, 3.4 bp of 1e6). Against the
  exact recursion `dcn|601` 3.51 → 0.15 bp and `dcn|default` 0.070 → 0.102 bp (an 801-point plan
  → the 400-point preset; away by 0.03 bp, inside its 1 bp bound); `lv_dcn` bit-identical; 3
  added.
- (d) local-vol DCN: `lv_dcn|default` and `lv_dcn|601`, pv up to 148.6 (`lv_dcn|601`). Against
  the exact recursion on the rows' constant 22 % surface 1.21 → 0.086 bp and 1.52 → 0.038 bp; 1
  added.
- (e) deletions: no value moves; the two declared-error rows
  `3b|{heston,slv}_snowball|half_step|stats` change text; 8 European-knock-in rows added
  (`3c|eki|*`).

Stage 3, the knock-in-side row: 721 rows, 0 unexpected diffs. The 15 rows of the evidence's r4
list equal `r4_book_gated.json` bit for bit, at most 1.11e-6 (`intraday_snowball`). The
`3c|eki` rows move in price by −4.14e-3 (Snowball), −3.25e-3 (Phoenix) and −8.14e-3 (reverse
Snowball), in bump vega by up to 8.91e-3, and the KO-reset by 6.8e-7 (a pre-maturity
statistic); each move was measured toward the independent reference (Fixed). The
`3c|eki|phoenix` row was rebuilt as a knock-in-at-maturity Phoenix; it had been the monthly one.

### Certification
The knock-in-side row moves 31 values in 10 certified cells, each equal bit for bit to its
pre-measured value (`r4_anchors_gated.json`); every other anchored value is unchanged and every
decision is unchanged. The six studies are amended and banked under
`docs/modelvalidation/certificates/{snowball-flat-bsm,phoenix-flat-bsm,ko-reset-flat-bsm,snowball-localvol-1d,adi2d-snowball-greeks,snowball-intraday-daily-ki-bsm}/2026-10-06/`,
their parents retired in place. The four schema-1 studies were amended with `--rerun-candidate`
on their PDE candidate; `adi2d-snowball-greeks` through the external-benchmark path (24 cells
carried bit for bit, 4 translated); the intraday study's reference re-ran value-identical.

| Study | Values / cells | Cells | Largest move |
|---|---|---|---|
| snowball-flat-bsm | 3 / 1 | `european_ki` | 4.33e-3 (pv) |
| snowball-localvol-1d | 12 / 4 | `calm_discrete_ki`, `calm_european_ki`, `crash_discrete_ki`, `crash_european_ki` | 0.165 contracts (`calm_european_ki` pv) |
| ko-reset-flat-bsm | 3 / 1 | `european_ki` | 5.4e-7 |
| snowball-intraday-daily-ki-bsm | 6 / 1 | `near_ko_on_ko_day_1s` | 5.4e-10 |
| phoenix-flat-bsm | 3 / 1 | `ki_stepdown` | 1.9e-15 |
| adi2d-snowball-greeks | 4 / 2 | `heston_pde\|near_expiry`, `heston_slv_pde\|near_expiry` (delta, gamma) | 1.8e-15 |

The local-vol European knock-in pv errors against the certificate's reference go +0.165 →
−0.0002 (calm) and +0.091 → −0.034 (crash) contracts; the flat `european_ki` pv goes from
+4.4e-3 to +4e-5 against the exact value. The adi2d moves are translated by the harness's own
interval arithmetic, shifting its intervals by at most 9.8e-14 contracts. Both `amend` changes
are listed under Added.

### Performance
Timing (`test/pde_bitwise/timing.py`, 0.9.0 and 0.10.0 interleaved, 15 repetitions per arm,
ARM64 macOS; cold medians in ms). A case whose 15 cold times spread by more than 10 % of the
median on either side is inconclusive, never a pass. The four statistics cases were
re-measured one family per process: their joint run spread by 24 to 81 %.

The untouched routes, gated at 1.05×: American 1.007×, barrier 1.013×, Snowball 1.007×,
Phoenix 1.018×, KO-reset 1.010×, local-vol Snowball 1.004×, Snowball statistics 1.015×,
Phoenix statistics 0.996×, KO-reset statistics 1.006×, local-vol Snowball statistics 1.006×:
none fails. The flat European is **inconclusive**: three runs gave 1.008×, 0.983× and 1.007×
with spreads of 10.8 to 28.5 % on a 23 ms single-shot case. Ungated: the memory and EXPIRY
Phoenix statistics, 1.005× and 1.001×.

The four routes moved onto the core are slower by design (the shared presets and the core's
time density, decisions 3 and 4); declared, not gated:

| Route (fast / standard / high) | `price` 0.9.0 | `price` 0.10.0 | `calculate_greeks` 0.9.0 | `calculate_greeks` 0.10.0 |
|---|---|---|---|---|
| Local-vol vanilla | 8.0 / 8.9 / 21.3 | 65.6 / 72.4 / 172 | 47.5 / 52.3 / 128 | 254 / 283 / 679 |
| Local-vol barrier, continuous | 9.4 / 10.3 / 24.4 | 71.1 / 69.0 / 166 | 28.0 / 30.7 / 73.3 | 70.4 / 68.2 / 167 |
| Local-vol barrier, daily | 12.6 / 13.7 / 31.0 | 112 / 122 / 288 | 37.8 / 41.0 / 92.6 | 112 / 123 / 239 |
| DCN | 30.6 / 39.4 / 42.1 | 90.5 / 103 / 219 | 76.5 / 98.0 / 122 | 90.5 / 103 / 218 |
| Local-vol DCN | 44.0 / 54.7 / 61.8 | 234 / 260 / 609 | 117 / 145 / 182 | 232 / 259 / 571 |

The double-barrier and touch products under EXPIRY build no grid (static replication).

### Known issues
- The analytic engines short-circuit a spot beyond a discretely monitored barrier although no
  observation is due: the double barrier returns the rebate (−0.086 on a monthly knock-out at
  spot 135), the single barrier the rebate or the vanilla (−0.202, +0.596, +0.032 on three
  monthly products); the double-sharkfin engine has the same test (read, not measured). At
  maturity the single-barrier analytic engine returns 0 for an untriggered knock-in's rebate,
  tests a schedule with no maturity record by the spot (5 for 35) and ignores a maturity
  record's own level (25 for 7). The PDE is right in all of these.
- Under `ACCUMULATE` a no-touch's interior records add 0, so only its maturity record ends it: a
  year out, records at 0.5 and 1, the PDE gives 4.304787 = `5 DF P[S_1 < 130]`, against 4.227537
  under first hit. Whether the mode has a meaning on a no-touch is a product question.
- `time_shift` drops a record that lands on the valuation instant, and when that empties the
  schedule it returns before shortening the maturity, so the product is not aged at all; a dated
  product valued on its exercise date raises.
- The double-barrier engine's Ikeda–Kunitomo option leg is fixed at 21 images: where the true
  value is below 1e-13 it returns 1.6e-5 (95 / 105, vol 30 %, 3 years) to 4.1e-4 (98 / 102, vol
  50 %, 1 year). It has no error control.
- The double sharkfin's continuous paid-at-hit leg integrates the survival curve with 48
  Gauss–Legendre nodes; next to a barrier (spot 129.9 on 80 / 130) the factor is off by 4.3e-7 to
  1.9e-6 on the exact probability.
- An explicit `GridConfig(bounds=...)` close to a barrier mis-states the far-field row: the
  autocallable row by −0.09 to −0.13 (−0.13 at 15 points below the knock-in barrier, against
  +4.3 before the knock-in-side row), the single-barrier row by 0.1 to 3.3, and the knocked-in
  upper row by −0.73 to −1.74 under a narrow upper bound. The right row there is a knock-in or
  knock-out probability; keep an explicit bound well beyond the barrier.

## [0.9.0] - 2026-10-04

PDE 1D event statistics (spec: `docs/superpowers/specs/2026-10-03-pde-1d-event-statistics-design.md`).
The Snowball, Phoenix and KO-reset statistics (and their local-vol variants) march in the
value's own groups: the value's boundary rows allocated per column, its θ rule, and the
adapter's event stages. The maturity leg is a marched column, so on every native 1D
route `pv = Σ knock-out cash + Σ coupon cash + maturity leg` to round-off and
`reconciliation_error` is a real number. Every stage was checked against the 510-row PDE
byte book with its declared movers. Prices move where a contract was mispriced: the
Phoenix release rule, the contractual coupon periods, the Phoenix knocked out at valuation
on a later observation, and the valuation-date decisions (a carried knock-in under
`disable_ko_after_ki`; a carried-knock-in KO-reset, which raised before). No certificate
identity moves.

### Added
- `AutocallableEventStats.reconciliation_error` is computed:
  `pv − Σ expected_discounted_cashflows`, round-off (≤ 1e-10) on native 1D routes.
- `AutocallableEventStats.provenance: StatisticsProvenance(statistics_model, price_model)`,
  `EventDistribution.provenance`, `EventDistribution.determination_times` (per maturity
  outcome; defaults to the last event time, so existing distributions are unchanged), and
  `StatisticsRefused` (a `PricingError`).
- Phoenix: `expected_discounted_coupon_cashflow` for INSTANT memory coupons and for
  EXPIRY coupons (per earning date, as QUAD V2). EXPIRY cash is also bucketed by payment
  event in the ledger, with `coupon_payment_is_path_dependent = True`.
- `KOResetEventStats`: `expected_discounted_pre_maturity_cashflow` /
  `…_post_maturity_cashflow`, `pre_maturity_probability` / `post_maturity_probability`,
  `pre_maturity_time` / `post_maturity_time`, their payment times, `pre_ko_payment_times` /
  `post_ko_payment_times` and `post_ko_times_are_offsets`. The PDE and QUAD V2 fill the
  buckets; legacy QUAD and Monte Carlo leave them `None`.
- `EventDistribution.from_autocallable_stats` accepts a `KOResetEventStats`.
- `KOResetSnowballPDESolver.calculate_event_stats` is native, and `streams` is honoured as
  on the Snowball.
- `PDEPlan.terminal["event_stats"]` can be `knocked_out_at_valuation`.
- The 2D Heston / SLV solvers' `statistics="refuse" | "flat_vol_surrogate"` option.
- `test/pde_statistics/`: an exact two-state reference for flat GBM, with its guards and
  a Monte Carlo qualifier, banked as JSON; a 32,001-point QUAD reference for KO-reset;
  the identity, column and stage tests.
- `PhoenixOption.active_ko_source_indices(env)`; `AccrualConfig.first_period_start` (where the
  first remaining coupon period started, recorded when an ageing drops an observation).

### Changed
- **Statistics policy.** Boundary rows, θ and event stages are the value's. Snowball
  probabilities move by ≤ 7.3e-8 on the byte book; the coupon-barrier smoothing moves
  Phoenix ones by ≤ 1.8e-6.
- **`expected_discounted_maturity_cashflow` is the marched leg**, not `pv − ΣKO cash`. It
  moves by the 0.8.0 residual (≤ 7.4e-6 on the byte book, 1.2e-4 on the study cases) and,
  on memory, EXPIRY and immediate-knock-out Phoenix, by the coupon cash it no longer holds.
- **2D statistics refuse by default.** `HestonSnowballPDESolver`,
  `HestonSLVSnowballPDESolver` and the Phoenix pair raise `StatisticsRefused` from
  `calculate_event_stats` / `price_with_events` unless built with
  `statistics="flat_vol_surrogate"`. That option returns 0.8.0's flat-vol 1D statistics,
  now with the 2D route check, the residual (the model difference, 0.26 on the book's
  Heston Snowball) and the provenance. The replay's `event_stats_fallback="mc"` no longer
  masks a refusal.
- **Phoenix release rule (decision 4).** A knock-out pays the memory arrears whatever
  the spot, plus the current coupon when its barrier is met, as QUAD V2 always did. This
  applies to the PDE value and statistics, the immediate-knock-out payoff,
  `decide_observations_at_valuation`, the lifecycle tracker and `PhoenixMCEngine`. A
  price moves only under memory with the coupon barrier above the knock-out barrier: on
  the study's dated product (coupon 105, knock-out 103) the PDE goes 97.811385 → 98.384223,
  against V2's 98.384358. Certified products are bit-identical.
- **Valuation-date decisions (decision 15).** The family's own
  `decide_observations_at_valuation`, with the carried knock-in state, decides a
  valuation-date observation on every entry point. The spot-versus-barrier shortcut is
  gone. A carried knock-in with `disable_ko_after_ki` no longer knocks out at valuation:
  the dated case moves 1.019178 → −0.674, against Monte Carlo's −0.664. A
  carried-knock-in KO-reset knocks out on its post schedule.
- `PhoenixOption.get_coupon_period_year_fractions` takes the pricing environment; a list
  or array argument raises `ValidationError` naming the 0.9.0 change. `accrual_factors` and
  a list coupon barrier are read by the observation's ORIGINAL schedule position, in the
  engines and in `decide_observations_at_valuation` (0.8.0 aligned them to the remaining
  records).
- An aged tenor Phoenix (no `initial_date`, a per-annum coupon) pays its first remaining
  period from the last dropped observation (`AccrualConfig.first_period_start`). Where that
  observation carries no time the period is unknown and a per-annum coupon raises
  `ValidationError` naming `fixed_coupon_year_fraction` and `accrual_factors`. A theta clone
  that crosses an observation is ex-coupon: on a 12 % tenor Phoenix one day before an
  observation theta is about −0.98 (the coupon paid); 0.8.0's −0.017 came from an
  over-accrued next period.
- `EventDistribution.from_autocallable_stats` raises `NotImplementedError` on a
  `KOResetEventStats` with `post_ko_times_are_offsets` (Monte Carlo under
  `PostKOScheduleMode.REBASED`) and `ValidationError` on one without its payment times.
- The legacy `ki_probability` on KO-reset PDE statistics is "settles knocked in", as on
  every PDE and QUAD route (the 0.8.0 QUAD delegation reported "ever"). The two explicit
  fields are unambiguous.

### Removed
| Removed | Replacement |
|---|---|
| `STATISTICS_LEGACY`, `GroupPolicy.zero_slope_after_events`, `GroupPolicy.dt_as_python_float`, `BoundaryPolicy.CARRY` | the `VALUE` policy; a rolling group supplies its rows |
| `StatisticsLegacy` and `core/statistics/legacy.py` | `core/statistics/group.py` and the families' groups |
| the Snowball / Phoenix statistics hooks (`_n_extra_event_cols`, `_set_extra_event_indicators`, `_extract_extra_event_stats`, `_t0_extra_indicator_overrides`) | the column layouts of `core/statistics/columns.py`, the adapter's stages, the valuation-date rules |
| the KO-reset solver's QUAD import and its `grid_points=501` delegation | native KO-reset statistics |
| `reconciliation_error = 0.0` hard-coded on the three PDE paths | the computed residual |
| the 2D solvers' silent inherited statistics | `statistics="flat_vol_surrogate"` |
| `_is_knocked_out_at_valuation` on the price and statistics paths | the family's `decide_observations_at_valuation` |
| `PhoenixOption.get_coupon_period_year_fractions(observation_times)` | `get_coupon_period_year_fractions(pricing_env)` |

### Fixed
- **`disable_ko_after_ki` statistics at maturity.** 0.8.0's Snowball statistics let
  knocked-in paths knock out on the final date. The last knock-out probability read
  4.06e-2 instead of 2.26e-3 (continuous knock-in) and 2.42e-2 instead of 4.69e-3
  (discrete), and `ki_survive_knocked_in_probability` was short by the same mass. Prices
  were right. Now within 2.1e-5 / 2.6e-6 (knock-out) and 1.9e-4 / 2.0e-5 (`ki_survive`)
  of QUAD V2 (`test/pde_statistics/test_disable_ko_after_ki.py`). Anyone who used 0.8.0
  (or earlier) PDE statistics on a `disable_ko_after_ki` Snowball for knock-out timing,
  expected life or the knocked-in loss probability should re-run them.
- **KO-reset statistics and conversion.** 0.8.0 ran a 501-point QUAD and folded
  `pde_pv − quad_pv` into the maturity leg. Pre-schedule knock-out probabilities were off
  by 3.8e-3 / 2.0e-3 / 1.4e-3 (spots 100 / 90 / 101) against a 32,001-point QUAD; native,
  they are within 1.3e-5 / 2.0e-5 / 6.9e-6, post-schedule within 3.8e-5, and the identity
  holds to 7e-14. The event distribution read only the pre schedule, carrying a maturity
  mass of 0.294 where 0.200 of the paths survive both. It now has both schedules as
  knock-outs and the two maturity outcomes at their own determination times.
- **Memory Phoenix coupon cash.** INSTANT memory coupons had no coupon cash; it sat in the
  maturity leg. On the study's product the coupon cash went from not reported to
  0.736235 (V2 0.736239) and the leg from −3.34829 to −4.084522 (V2 −4.084004); the identity holds to 9e-16.
- **EXPIRY Phoenix coupon cash.** Deferred coupons are reported per earning date and
  bucketed at each termination's payment time, with the path-dependent flag (no cash
  leg reads the buckets yet; below).
- **Knocked-out distribution.** `price_with_events` on a product knocked out at valuation
  returned `{MATURITY_NO_KO: 1.0}` at maturity, so a cash leg treated the note as alive.
  It now returns the knock-out at its settlement, the same as `calculate_event_stats`.
- **Phoenix knock-out coupon.** On a Phoenix knocked out at valuation, the coupon paid with
  the knock-out (1.019178 on the byte book) was in the maturity leg; it is now coupon cash
  and the leg is 0, as QUAD V2. The price and the statistics took the FIRST period's coupon
  and barrier whatever the observation: on a dated Phoenix knocked out on its second
  observation the PDE paid 1.019178 where the contract pays 1.052055, and with a list
  barrier not met at that observation it paid a coupon the contract does not. Both now take
  the product's own decision (`test_a_later_observation_knock_out_pays_the_decided_coupon`).
- **Lifecycle-knocked-in `ki_survive`.** A product carrying the lifecycle knock-in flag
  reported `ki_survive_knocked_in_probability = 1` although most paths may knock out
  first. It is now the marched value: 0.277176 on the measured Snowball (1 − ΣKO =
  0.277176). With the knock-in streams pruned it reads 0.0, like every pruned field.
- **Contractual coupon periods (decision 13).** A dated Phoenix valued after its start
  gave its first remaining period the fraction from the valuation date (0 on an
  observation date) in the PDE and Monte Carlo, while QUAD V2 used the contract. Byte
  book: observation-date pay +1.019178 and miss +0.7581, ten days in +0.4596 (101.683759 and
  86.114188 against V2's 101.684019 and 86.114162). The deterministic control is exact.
  Tenor Phoenix theta moves by about one day of coupon, because 0.8.0's theta clone
  truncated the first period: up to 1.0e-2 on the byte book, 3.3e-2 on a 12 % tenor
  Phoenix (−0.01685 → +0.01595).
- **Valuation-date regimes.** A carried knock-in's regime decides (above).
- **Dated KO-reset `KeyError`.** A dated KO-reset valued on its first observation date
  raised `KeyError` (the post observation was not a grid node); both schedules' times are
  now in the grid request.

### Known issues
- EXPIRY coupon cash reaches no cash leg: `price_with_events` carries no ledger, and a leg
  that needs path-dependent coupon timing raises as it does for QUAD V2.
- `ki_ever` under continuous knock-in carries the first-passage correction's error
  (1e-4 to 9e-4) and does not converge under time-only refinement.
- `ki_ever` counts a breach after a knock-out, on every engine (a definition).
- The value's far-field rows ignore the certain far-edge knock-in and knock-out, and the
  Phoenix upper row omits the next coupon. None of this is visible at the spot on the
  measured cases.
- The 2D statistics are the flat-vol model's (surrogate only; native 2D is a later spec).
- KO-reset maturity buckets against QUAD V2: the PDE's `M_pre` is −1.5e-2 (cash) from V2's
  at spot 100 (−2.0e-2 at `high`; about 1.5e-4 in probability); the gate is 3e-2.

### Per-stage deltas (PDE byte book, 510 rows)
- (a) Snowball value policy: 97 rows. Leg ≤ 7.4e-6, probabilities ≤ 7.3e-8, except
  `disable_ko_after_ki` (leg 3.45, `ki_survive` 3.1e-2: the fix above) and the 2D
  surrogate residual (0.26).
- (b) Phoenix value policy: 60 rows. Coupon cash out of the leg (up to 6.49 on the
  observation-date memory row), probabilities ≤ 1.8e-6, 2D surrogate residual 0.36.
- (c) native KO-reset: 33 rows. Probabilities up to 1.4e-1 on the legacy `ki_probability`
  (its definition, above), post-schedule cash 5.7e-2, distribution payment times.
- (d) valuation-date rules: 35 rows. `dkak_carried` price 1.693, the lifecycle
  `ki_survive` (up to 0.96), the KO-reset immediate knock-out.
- (e) 2D refusal: 3 rows, now `StatisticsRefused`.
- (f) release rule: 9 rows. `phoenix_c105` +0.5728, `lu_fallback|phoenix` +6.1e-4 (one
  dual cell spans both barriers on that 4-node grid).
- (g) contractual periods: 63 rows. Dated Phoenix prices up to +1.019178, tenor Phoenix
  theta ≤ 1.0e-2, `phoenix_c105` ≤ 1.3e-13.

Goldens: the replay event frames (`scalar_bsm`, `localvol`) and the phase-4 execution
event distributions were regenerated once (`test/replay_golden/README.md`,
`test/execution/goldens/PROVENANCE.md`); the `book` (QUAD) frames are byte-identical.

### Performance
- The statistics block skips its all-zero columns (an observation the backward march has
  not reached) and solves them once as a +0.0 template. The tridiagonal solve calls LAPACK
  `gtsv` directly, without scipy's argument validation. The first-passage correction forms
  only its one row on a block. The value rows are reused, not recomputed. The KO-reset
  terminal resolves the pre-knock-in tenor once per surface, not per node. All are
  bit-identical on the byte book.

Timing gate (`test/pde_bitwise/timing.py`, 0.8.0 and 0.9.0 interleaved, ARM64 macOS with
the `accel` extra; median ms over 15 rounds). The price families come from the full run
(IQR ≤ 3 %). The statistics cases come from a rerun of the gated four alone
(`--families`, three repetitions per process, IQR ≤ 8 %): the full run was inconclusive,
with bimodal efficiency-core times on a loaded machine.

| Case | Warm 0.8.0 | Warm 0.9.0 | Ratio | Cold ratio | Gate |
|---|---|---|---|---|---|
| Snowball price | 42.9 | 34.9 | 0.81× | 0.87× | |
| Phoenix price (memory) | 652 | 455 | 0.70× | 0.71× | |
| KO-reset price | 25.5 | 17.3 | 0.68× | 0.69× | |
| Local-vol Snowball price | 117 | 107 | 0.91× | 0.93× | |
| European / American / barrier price | | | 1.00× | 1.00× | |
| Snowball statistics | 159 | 146 | 0.92× | 0.94× | pass |
| Local-vol Snowball statistics | 248 | 225 | 0.91× | 0.91× | pass |
| Phoenix statistics (no memory, INSTANT) | 230 | 247 | **1.075×** | 1.06× | **missed** (1.05× warm) |
| KO-reset statistics | 40 | 72 | **1.79×** | 1.78× | **not comparable** (0.8.0 was a 501-point QUAD) |
| Phoenix statistics (memory) | 877 | 1,279 | 1.46× | | recorded |
| Phoenix statistics (EXPIRY, memory) | 972 | 2,348 | 2.42× | | recorded |

- **Phoenix statistics miss the 1.05× gate by 2.5 points (+17 ms).** The coupon cash is a
  marched column (spec §5.3) where 0.8.0 multiplied the trigger probability. That is 39
  columns per surface instead of 26; the solve cost is linear in the column count, and
  the faster solve absorbs most but not all of it.
- **KO-reset statistics compare two algorithms.** 0.8.0 ran a 501-point QUAD next to the PDE
  price, with probability errors of up to 3.8e-3. 0.9.0 marches 15 columns per surface on
  the PDE, which cannot cost less than the PDE price it extends.
- Memory and EXPIRY Phoenix carry 183 and 365 columns per surface (12 observations, 12
  memory states); 0.8.0 reported no coupon cash for them. Pruning `streams` does not reduce
  the count.

## [0.8.0] - 2026-10-03

PDE 1D numerical policy (spec: `docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`).
Seven deliberate numerics changes, each its own commit checked against the 445-row PDE byte
book with its declared movers: the grid day is the environment's business days, the step
cap raises instead of scaling, the barrier family projects discrete observations, Snowball
`NODAL` means nodal, Phoenix takes θ = 1 on zero-diffusion steps, half-step damping where
monitoring is discrete, and forward far-field rows. No flip touches the damping of a
product with continuous knock-in: on a 252-day environment, inside 0.7.0's 5,000-step cap
and with the default projection, it keeps its 0.7.0 numbers bit for bit (one exception,
below); the grid day (a) and the cap (b) move it like any product. The flat Snowball, Phoenix and
KO-reset and the local-vol Snowball certificates are amended and re-banked (2026-10-02) with
their PDE candidates re-admitted; the intraday daily-KI certificate is re-banked (2026-10-03)
with QUAD V2 admitted and its PDE candidate rejected, as in 0.7.0.

### Added
- `CrankNicolson(damping="full_step" | "half_step")`: the damping form; `None` is the
  route's. Half-step splits each damped interval at its midpoint; both halves take
  `event_theta` below an event and 1.0 at the terminal node.
- `event_projection` is honoured by `BarrierPDESolver`, `DoubleBarrierPDESolver`,
  `OneTouchPDESolver` and `DoubleOneTouchPDESolver`.
- `MarketSnapshot.bus_days_in_year`, `GridRequest.day_count`, `GridRequest.refinement`,
  `TimeRefinement`, `time_fill`, `EventRecord`, `DiscreteEvent`, `apply_discrete_event`
  (`quantark.asset.equity.engine.pde.grid`).
- `DAMPING_DEFAULTS["core-continuous-ki"]`, `["2D"]`, `["local-vol"]` (`pde.grid.config`).
- `grid.mesh_kernels.backend()`: `"numba"` with the `accel` extra, else `"python"`.
- `pde_declared.environment_day_count` / `refuse_case_day_count` (model validation): the
  environment's business days recorded on a PDE candidate's `grid` record.

### Changed (defaults)
- **Core damping:** half-step, one interval at the terminal node and at each event,
  `(half_step, 1, 1)`, for every product without continuous knock-in; a product with
  continuous knock-in keeps full-step `(1, 2)`. The class is read from the contract, never
  from the spot or the knocked-in state.
- **Base-family discrete observations:** dual-cell projection (`event_projection=None` is
  `CELL_AVERAGE` on every core route). `"nodal"` keeps the 0.7.0 masks.
- **`max_steps`:** 12,000; a grid over it raises `ValidationError` naming the count it
  needs, `GridConfig(max_steps=...)` and `steps_per_day`. A grid is never scaled.
- **Snowball `NODAL`:** nodal on interior KO and discrete-KI events too.
- **Phoenix:** θ = 1 on zero-diffusion steps, as every value group.
- **Grid day:** `steps_per_day` counts the pricing environment's `bus_days_in_year`.
- **Far-field rows:** European and Barrier puts' lower row is `K·DF(t,T) − S_min·CF(t,T)`
  (floored at 0); the Barrier call's upper-row carry factor is the forward one
  `exp(−(q(T)T − q(t)t))` (it moves only on a term-dependent carry curve).
- `adapter.stepper(params)` becomes `adapter.stepper(params, product)`.
- `GridRequest(day_count=...)` and `MarketSnapshot(bus_days_in_year=...)` are required: code
  that constructs either directly gets a `TypeError` until it passes them.
- Event statistics (KO/KI probabilities) march the same layout, so they move with the half
  nodes at (f), as the prices do; their conventions are unchanged.
- The step cap holds a little under six years at `high` and twelve at `standard` on a 252-day
  environment; a six-year monthly Snowball at `high` now raises where 0.7.0 scaled it.
- The certificates' `grid` record: `max_steps` 12000, `day_count` from the environment,
  and `discrete_monitoring_damping: ["half_step", 1, 1]`. A full-step `(1, 2)`
  configuration with `max_steps=5000` on a 252-day environment still serializes to the
  0.7.0 record.
- Prepared PDE sessions: artefact builder `pde-prep/v3` (the request carries `refinement`
  and `day_count`); no 0.7.0 artefact is reused.

**Clean break (count meaning).** A caller who sets a count and no form, for example
`CrankNicolson(event_damping_steps=2)`, gets two halved intervals after flip (f) on every
solve that resolves against `"core"`, where 0.7.0 gave two whole steps; on a product with
continuous knock-in the count keeps its 0.7.0 meaning. Nothing can detect the intent, so no
error is possible. `CrankNicolson(damping="full_step", terminal_damping_steps=1,
event_damping_steps=2)` restores 0.7.0 everywhere; `damping="full_step"` alone takes the
route's counts, `(full_step, 1, 1)` on the core. A `CrankNicolson` **subclass** is a custom
stepper, used as given: one that sets counts and no `damping` priced in 0.7.0 and now raises
`ValidationError` naming the fields to set.

### Removed
| Removed | Replacement |
|---|---|
| `GridConfig(day_count=...)` (raises `ValidationError`) | `PricingEnvironment(bus_days_in_year=...)` |
| `TimeLayout.fill_scaled`, `TimeLayout.cap_exceeded`; the same keys in `explain` and the intraday numerical record; `LifeSurfaceSolution.fill_scaled` | none: a grid over `max_steps` raises |
| `groups.VALUE_NO_ZERO_DIFFUSION_OVERRIDE` | `groups.VALUE` |

`GridConfig` has eight fields. `TimeLayout` and `LifeSurfaceSolution` are results the
library constructs, so reading a removed field is a plain `AttributeError`.

### Fixed
- **Put far-field row.** The European and Barrier put's lower row was its `S → 0` limit
  `K·DF` at a grid edge four standard deviations below spot, ~39 too high there. Error
  against Black-Scholes at 400 points (pv / delta / gamma), study environment:

  | European put, strike 100 | 0.7.0 | 0.8.0 | the call, for scale |
  |---|---|---|---|
  | vol 22%, 1y, spot 100 | +2.5e-3 / −4.3e-4 / +9.1e-5 | +1.1e-4 / +4.1e-5 / +2.1e-6 | +2.1e-4 / +4.2e-5 / +2.2e-6 |
  | vol 22%, 1y, spot 80 | +1.9e-3 / −5.4e-4 / +1.1e-4 | −1.4e-5 / −6.6e-5 / +1.7e-6 | +2.6e-5 / −6.6e-5 / +1.7e-6 |
  | vol 22%, 3m, spot 100 | +4.2e-3 / −1.4e-3 / +5.9e-4 | +1.9e-4 / +8.4e-5 / +2.2e-5 | +2.5e-4 / +8.4e-5 / +2.2e-5 |
  | vol 22%, 3y, spot 100 | +1.1e-3 / −1.0e-4 / +1.5e-5 | −7.6e-5 / +2.3e-5 / +1.9e-7 | +2.0e-5 / +2.7e-5 / +2.2e-7 |
  | vol 12%, 1y, spot 100 | +3.9e-3 / −1.2e-3 / +4.6e-4 | +2.0e-4 / +6.2e-5 / +1.6e-5 | +2.8e-4 / +6.2e-5 / +1.6e-5 |
  | vol 40%, 1y, spot 100 | +1.0e-3 / −9.3e-5 / +1.3e-5 | −9.6e-5 / +2.4e-5 / +1.2e-7 | +1.4e-5 / +2.9e-5 / +1.6e-7 |

  The down-and-in put, dominated by that row, now meets its exact reference. The American
  put's lower row is unchanged (spec §9).
- **Snowball `event_projection="nodal"`** projected interior events anyway; it now masks
  them (spec §4.6 cases, NODAL pv 0.7.0 → 0.8.0: ordinary +3.5e-2, spot 102.5 +2.6e-2,
  monthly discrete KI +2.6e-2).
- **Phoenix** took θ = 0.5 on zero-diffusion (trading-clock night and weekend) steps where
  every other family took 1 (trading-clock Phoenix: +1.6e-7).
- **Continuous knock-in, alive boundary:** the knocked-in boundary rows are written before
  the alive rows that read them. Bitwise on every certified cell but one:
  `snowball-localvol-1d/calm_near_expiry` moves by 1.1e-13 (pv) to 3.5e-11 (delta) relative;
  the sparse-LU fallback Snowball (two interior nodes) moves +1.457.

### Performance
- **Mesh build.** The concentrated mesh's RK4 integration (the cold-solve cost, spec §4.9)
  runs on a Python-float kernel, or on a Numba kernel with the `accel` extra
  (`pip install "quantark[accel]"`), both bit-identical to the 0.7.0 NumPy reference on 549
  layouts. A value a kernel does not handle (non-finite, zero denominator, more than eight
  critical points) re-runs that call on the NumPy reference, so every exception and fallback
  is 0.7.0's. On 49 production layouts: 43.1 s (NumPy), 4.06 s (Python kernel), 0.30 s (Numba).
  CI requires the Python kernel on its four Python jobs and Numba in the `accel` job
  (`QUANTARK_REQUIRE_MESH_KERNEL`, a test switch). The
  Numba runtime adds ~30 MB to a process's peak memory.
- **Barrier step events** fold in place on the grid column (no per-step copy), and the
  terminal settlement timing is resolved once per march, not once per step (bitwise).

Timing gate (`test/pde_bitwise/timing.py`, 0.7.0 and 0.8.0 interleaved, two rounds of five
repetitions, ARM64 macOS with the `accel` extra; seconds, better round):

| Family (timing book, `standard`) | Cold 0.7.0 | Cold 0.8.0 | Speed-up | Warm 0.7.0 | Warm 0.8.0 |
|---|---|---|---|---|---|
| European | 0.774 | 0.022 | 35× | 0.0185 | 0.0150 |
| American | 0.765 | 0.026 | 29× | 0.0207 | 0.0207 |
| Snowball | 0.916 | 0.050 | 18× | 0.0434 | 0.0432 |
| Local-vol Snowball | 1.002 | 0.126 | 8.0× | 0.119 | 0.119 |
| Event statistics | 1.04 | 0.165 | 6.3× | 0.158 | 0.158 |
| Phoenix (memory) | 1.542 | 0.668 | 2.3× | 0.654 | 0.658 |
| Barrier (continuous) | 0.030 | 0.021 | 1.4× | 0.0282 | 0.0189 |
| KO-reset | 0.026 | 0.026 | 1.0× | 0.0254 | 0.0255 |

A cold solve is now dominated by its march: the Phoenix's mesh share fell from 0.89 s to
0.01 s, and the barrier and KO-reset cases build no concentrated mesh. Event statistics
from 15-repetition interleaved runs. Peak memory rises ~30 MB with Numba loaded.

### Per-flip deltas (PDE byte book, largest price moves)
- (a) grid day: 244-day environments only — Phoenix memory −2.7e-4, continuous-KI Snowball
  −1.6e-4, European −5.2e-8, intraday Snowball +6.4e-8. The intraday route's grid moved
  because its numerical environment has 244 days (F1).
- (b) cap: five-year standard Snowball −4.9e-7 (5,040 steps, was 4,980), three-year high
  −2.1e-6 (6,048, was 4,968).
- (c) projection: discrete barrier calls up to ±3.8e-2, double knock-out −2.8e-2, the
  down-and-in composite −8.0e-3, touches ~1e-3.
- (d) Snowball NODAL: +1.2e-2 to +1.8e-2 on the book's NODAL rows.
- (e) Phoenix θ: +1.6e-7 on the trading-clock Phoenix only.
- (f) half-step damping: local-vol Phoenix daily KI −3.6e-3, Snowball daily KI −3.3e-3,
  KO-reset discrete −1.6e-3, Snowball monthly −1.3e-4, discrete Snowball/Phoenix ≤ 9e-5,
  barrier/touch ≤ 3.6e-5, European ~1e-6. No continuous-KI row moved.
- (g) far field: puts (above); flat-curve barrier calls bitwise unchanged.

Goldens: replay frames (scalar_bsm, localvol) moved at (f) by ~1.6e-5 of notional;
execution goldens at (d) and (f) (`test/execution/goldens/PROVENANCE.md`).

## [0.7.0] - 2026-10-01

PDE 1D one core (spec: `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`;
architecture: `docs/pde-core.md`). Every shared-grid 1D PDE family (European, American,
the barrier family, Snowball, Phoenix, KO-reset, and the local-vol Snowball and Phoenix)
prices through one `prepare → march → read` path; the four per-family time-stepping loops
and the separate statistics sweep are gone. Numbers are unchanged: a 229-row `float.hex`
byte compare of the 1D book against 0.6.0 shows zero differences, and every banked
certificate keeps its identity. The scheme's settings move onto a stepper component, a
clean break with the old names raising `ValidationError` naming their replacement.

### Added
- `PDEParams.stepper`, `PDEParams.operator`, `PDEParams.readout`: component slots whose
  built-ins are `CrankNicolson()`, `CentralDifference()` and `StandardReadout()`
  (`quantark.asset.equity.engine.pde.core.components`). `CrankNicolson` carries the
  scheme's four settings: `theta`, `event_theta`, `terminal_damping_steps`,
  `event_damping_steps` (`None` = the route's defaults). A custom component runs on the
  core routes; the core validates its results and raises `NumericalError` naming it.
  `from_config` accepts `stepper: {theta: 1.0, ...}` and rejects `operator` / `readout`.
- `explain(product, pricing_env)` on `PDEEngine` and every core solver: the resolved plan
  (route, settings with their source and applicability, space, time, events, scheme,
  outputs, identities, per-operation terminal outcomes) from the same preparation `price`
  runs, without marching.
- `resolved_grid_evidence(params)` (`pde.grid.config`): the certificate `grid` record,
  the nine resolved grid fields plus the stepper's two damping counts under their 0.6.0
  keys. `PDEParams.evidence()` for `engine_config`.
- `OutputRequest`: what one call wants from a preparation (price, Greeks, event
  statistics, grid, life surface).
- `Calendar.identity()`: a value identity (name, sorted holidays, weekend days) for the
  settlement-calendar part of the schedule caches' keys.

### Removed
| Removed name | Replacement |
|---|---|
| `PDEParams(theta=...)` | `stepper=CrankNicolson(theta=...)`; 0.5 is Crank-Nicolson, 1.0 backward Euler |
| `PDEParams(event_theta=...)` | `stepper=CrankNicolson(event_theta=...)` |
| `GridConfig(terminal_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(terminal_damping_steps=N))`; 0 disables |
| `GridConfig(event_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(event_damping_steps=N))`; 0 disables |

`GridConfig` has nine fields and raises `ValidationError` (not `TypeError`) for the two
removed ones. The 0.6.0 entries that pointed at `GridConfig` damping counts
(`use_rannacher`, `rannacher_steps`, `rannacher_at_events`, `event_rannacher_steps`,
`DCNPDEEngine(rannacher_steps=)`) now point at the stepper. `TimeLayout` no longer carries
the damping index sets; `CrankNicolson.schedule` computes them.

### Changed
- The factorization caches key on the exact θ. Two θ agreeing to 6 (LU) or 12 (banded)
  decimals no longer share a matrix. `dt` stays rounded to 12 places, as every banked
  number was computed.
- Event statistics are a column group of the price march. `calculate_event_stats`
  no longer re-prices, and `price_with_events` marches once.
- Prepared sessions store the bound `Layout` (artefact builder `pde-prep/v2`). Bump
  contexts are fresh instances holding a frozen layout; the transient-attribute copy
  machinery is gone (the 2D Heston/SLV Greeks keep a legacy clone).
- Market memos are keyed by value fingerprints, so mutating a market object after a solve
  is no longer silently stale; an unfingerprintable input is sampled fresh.
- Certificate candidates are built from their declared params (`PDEParams.from_config`),
  so a declared setting is the one executed; a key the engine rejects fails the study at
  load.
- The DCN, local-vol vanilla/barrier and 2D Heston/SLV solvers read their damping counts
  and weights from `params.stepper`, and reject a custom component ("no core seam until
  sub-project 3"). `LocalVolPDESolver` checks route applicability on construction.

### Fixed
- Event statistics on a reused solver no longer reuse another market's factorizations
  (measured: a 0.041 KO-probability error on the second market).
- American Greeks on a fresh solver, for exercise with a settlement lag, no longer raise.
- PDE solvers deep-copy and pickle after pricing.
- A session no longer stores a factorization for a stepper without a value fingerprint
  (two such steppers could have shared one).
- Knock-in barriers (single and double) priced through a `PricingSession` now equal the
  direct price bit for bit. 0.6.0 marched the knock-out leg on the knock-in product's own
  spatial grid, masking the barrier instead of using the hard-edged domain (measured:
  up-and-in call −0.9%, double knock-in −2.2%). A session's layout is now refused, not
  reassembled, when the hard bounds differ. The session path was not in the byte book;
  it now has an execution test.
- A session with `cache_enabled=False` no longer warms a factorization pack.

Unchanged and still raising: bump-context Greeks on a knock-in barrier that has not
knocked in (`PricingError`: the frozen layout has the knock-in's bounds, the knock-out
leg needs the barrier's). It is a declared failure in the byte book; freezing the
knock-out leg's layout is sub-project 3 work.

### Sub-project 3
Still planned: the environment's business day as the grid day (full re-certification);
half-step damping as a `CrankNicolson` option; coefficient-break nodes; a budget default
for path-independent routes; failing instead of scaling at the caps; DCN and the
local-vol vanilla/barrier space axis onto the core; native KO-reset statistics. Added by
this release, each its own re-certified commit: statistics Level 2 (the statistics group
on the value policy with a marched maturity leg, memory-state Phoenix coupons); the
zero-diffusion θ override for Phoenix; Snowball `NODAL` on interior schedule events (or
withdrawing `NODAL` from the shared-grid table); native 2D Heston/SLV statistics.

## [0.6.0] - 2026-09-29

PDE 1D surface cleanup (spec: `docs/superpowers/specs/2026-09-28-pde-1d-surface-cleanup-design.md`).
Every 1D PDE route — the shared-grid solvers behind `PDEEngine`, the local-vol vanilla
and barrier solvers, and DCN — is configured by one `PDEParams` and nothing else, and
nothing is silently ignored: a field a route cannot honour raises. This is a clean break:
the removed names raise `ValidationError` naming their replacement. Numbers are unchanged
on every route except the local-vol discretely monitored barrier (see Changed), proven by
a 112-row `float.hex` byte compare of the whole 1D book against 0.5.0.

### Added
- `GridConfig.time_steps`: a total time-step budget over the horizon, spread over the
  mandatory intervals by length (at least one step each), and the alternative to
  `steps_per_day`. Exactly one time basis is in force after resolution; an overlay that
  sets one basis clears the other, and setting both raises.
- `GridConfig.evidence()` (the certificate serialization: unset optionals omitted, so no
  banked identity moves), `GridConfig.with_steps_per_day()` and
  `GridConfig.with_time_steps()`.
- `PDEParams.resolved_grid()`: the requested grid (the `accuracy` preset under the `grid`
  overlay), before the `max_steps` cap that binds once a maturity is known.
- The route applicability table (`quantark.asset.equity.engine.pde.route_applicability`,
  `ROUTE_RULES`): per route, the fields it cannot honour. An inapplicable field accepts its
  preset default and rejects an explicit override; a rejected field rejects any explicit
  value. The error names the field, the route, and sub-project 3.
- `PDEParams.from_config(config_or_path, **overrides)` over field names, with `grid` a
  nested `GridConfig` mapping (a `bounds` list becomes a tuple).

### Removed
`PDEParams`, `PDEEngine` and the DCN constructors take keyword arguments only; each
removed name raises `ValidationError` with its replacement:

| Removed | Replacement |
|---|---|
| `PDEParams(grid_size=...)` | grid=GridConfig(points=...) on every route. |
| `PDEParams(time_steps=...)` | grid=GridConfig(time_steps=N): the same meaning, a total step budget over the horizon. steps_per_day= is the per-day alternative. |
| `PDEParams(s_max=...)` | grid=GridConfig(bounds=(None, hi)) on the shared-grid solvers. On the two local-vol solvers an explicit bound is rejected until sub-project 3: the kernel sets its own domain and aligns the top, so there is no hard edge to honour. |
| `PDEParams(s_min=...)` | removed; it was read by nothing. |
| `PDEParams(use_rannacher=...)` | grid=GridConfig(terminal_damping_steps=N); 0 disables. use_rannacher=False also switched off event damping on the 2D Heston/SLV solvers; add event_damping_steps=0 to keep that |
| `PDEParams(rannacher_steps=...)` | grid=GridConfig(terminal_damping_steps=N); 0 disables |
| `PDEParams(rannacher_at_events=...)` | grid=GridConfig(event_damping_steps=N); 0 disables |
| `PDEParams(event_rannacher_steps=...)` | grid=GridConfig(event_damping_steps=N); 0 disables |
| `PDEParams(boundary_mode=...)` | removed; asymptotic boundaries are the only treatment |
| `PDEParams(continuous_ki_correction=...)` | removed; the first-passage correction always applies to continuously monitored KI |
| `PDEParams(ki_monitoring_mode=...)` | removed from the PDE; each barrier is monitored as the product declares it (discrete dates, continuous, or maturity-only). To price a dense discrete KI as continuous, declare it continuous on the product. QuadParams.ki_monitoring_mode is unchanged. |
| `PDEParams(cache_strategy=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(grid_cache_max_entries=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(banded_cache_max_entries=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(use_banded_solver=...)` | removed; the linear solver is chosen by the solver family |

| Removed | Replacement |
|---|---|
| `PDEEngine(method=...)` | Crank-Nicolson is the only built-in scheme; PDEEngine(params=PDEParams(...)). theta=1.0 gives backward Euler. |

| Removed | Replacement |
|---|---|
| `DCNPDEEngine / LocalVolDCNPDEEngine(num_space_nodes=...)` | DCNPDEEngine(params=PDEParams(grid=GridConfig(points=...))) |
| `DCNPDEEngine / LocalVolDCNPDEEngine(rannacher_steps=...)` | grid=GridConfig(terminal_damping_steps=N, event_damping_steps=N) (DCN applies both in its half-step form) |
| `DCNPDEEngine / LocalVolDCNPDEEngine(s_min_mult=...)` | removed; never read |
| `DCNPDEEngine / LocalVolDCNPDEEngine(s_max_mult=...)` | removed; never read |
| `DCNPDEEngine / LocalVolDCNPDEEngine(concentration=...)` | removed; never read |

Also removed: `BackwardOperator` (`backward_operator.py`), `ContinuousKICorrection`,
`make_pde_params`, `PDE_PARAM_PRESETS`, the `"pde"` branch of `make_engine_params` /
`list_param_profiles` (PDE has no profile tier: use `PDEParams(accuracy=...)`), the
class-level cache setters (`set_cache_enabled`, `set_cache_strategy`,
`set_grid_cache_max_entries`, `clear_grid_cache`), and `reject_legacy_resolution_knobs`.

### Changed
- **Local-vol vanilla and barrier solvers are configured by `PDEParams`.** Previously
  `accuracy` and `grid` were ignored there; they are now honoured or rejected per the
  route table. Their own preset table keeps today's plan at `standard` (400 points, a
  200-step budget); `fast`/`high` are new resolution choices. `theta` reaches both
  kernels; the vanilla's terminal damping is 0 or 1 (an explicit count of 2 or more is
  rejected). Bitwise at the defaults and for continuous and maturity-only monitoring.
- **The local-vol discretely monitored barrier observes on exact event nodes** (the
  declared number change). Each interior observation date is a node of the time layout,
  the step budget spread over the intervals between them, instead of the nearest node of
  a uniform grid. Byte gate: the two `lv_barrier_discrete` rows move by -7.9e-4 and
  -1.0e-3. Against a 1600 x 2400 reference PDE (every date on a node), flat 20% vol:

  | product | grid | base PDE | branch PDE | abs(base-ref) | abs(branch-ref) |
  |---|---|---|---|---|---|
  | UO call B130 monthly | 400x200 | 3.92225 | 3.92154 | 2.47e-03 | 1.75e-03 |
  | UO call B130 monthly | 600x240 | 3.92049 | 3.92049 | 7.03e-04 | 7.03e-04 |
  | UO call B130 monthly | 200x200 | 3.92822 | 3.92752 | 8.44e-03 | 7.74e-03 |
  | UO call B130 monthly | 800x400 | 3.91984 | 3.92013 | 5.49e-05 | 3.44e-04 |
  | UO call B130 monthly | 400x150 | 3.92460 | 3.92149 | 4.81e-03 | 1.70e-03 |
  | UO call B120 monthly | 400x200 | 1.76249 | 1.76185 | 3.05e-03 | 2.41e-03 |
  | UO call B120 monthly | 600x240 | 1.76040 | 1.76040 | 9.57e-04 | 9.57e-04 |
  | UO call B120 monthly | 200x200 | 1.77063 | 1.77001 | 1.12e-02 | 1.06e-02 |
  | UO call B120 monthly | 800x400 | 1.75965 | 1.75991 | 2.13e-04 | 4.74e-04 |
  | UO call B120 monthly | 400x150 | 1.76428 | 1.76179 | 4.84e-03 | 2.35e-03 |
  | UO call B120 quarterly | 400x200 | 2.16190 | 2.16190 | 1.04e-03 | 1.04e-03 |
  | UO call B120 quarterly | 600x240 | 2.16128 | 2.16128 | 4.18e-04 | 4.18e-04 |
  | UO call B120 quarterly | 200x200 | 2.16532 | 2.16532 | 4.46e-03 | 4.46e-03 |
  | UO call B120 quarterly | 800x400 | 2.16107 | 2.16107 | 2.06e-04 | 2.06e-04 |
  | UO call B120 quarterly | 400x150 | 2.16442 | 2.16189 | 3.55e-03 | 1.03e-03 |
  | DO put B85 monthly | 400x200 | 1.20049 | 1.20002 | 4.22e-03 | 3.75e-03 |
  | DO put B85 monthly | 600x240 | 1.19755 | 1.19755 | 1.27e-03 | 1.27e-03 |
  | DO put B85 monthly | 200x200 | 1.21140 | 1.21096 | 1.51e-02 | 1.47e-02 |
  | DO put B85 monthly | 800x400 | 1.19682 | 1.19701 | 5.46e-04 | 7.37e-04 |
  | DO put B85 monthly | 400x150 | 1.20189 | 1.19997 | 5.61e-03 | 3.69e-03 |
  | UI call B130 monthly | 400x200 | 4.90751 | 4.90822 | 1.83e-04 | 5.31e-04 |
  | UI call B130 monthly | 600x240 | 4.90791 | 4.90791 | 2.23e-04 | 2.23e-04 |
  | UI call B130 monthly | 200x200 | 4.90897 | 4.90967 | 1.28e-03 | 1.98e-03 |
  | UI call B130 monthly | 800x400 | 4.90809 | 4.90780 | 4.00e-04 | 1.11e-04 |
  | UI call B130 monthly | 400x150 | 4.90516 | 4.90827 | 2.53e-03 | 5.81e-04 |

  Closer to the reference in 16 of 25 cells; where the old grid misplaced a date by up to
  half a step (400x150) the error falls 2-3x, and elsewhere both placements sit inside
  the grid's own error band (every cell moves by <= 7e-4).
- **DCN is configured by `PDEParams`** with its own `accuracy` presets, which scale
  spatial resolution only (401 / 801 / 1601 points; `fast` and `high` have no banked
  evidence). The observation schedule is the time resolution: an explicit
  `steps_per_day` or `time_steps` is rejected, as is an explicit CELL_AVERAGE. The damping
  counts drive DCN's half-step form; `theta` and `event_theta` reach the loop. Bitwise at
  the defaults.
- **`ki_monitoring_mode` is removed from the PDE.** Each barrier is monitored as the
  product declares it. The two honest alternatives: `QuadParams.ki_monitoring_mode`
  (BGK stays in QUAD), or declare the KI continuous on the product.
- **`event_projection` defaults to `None`**, the route's own treatment (CELL_AVERAGE on
  the shared-grid autocallable solvers, as before).
- **`PDEEngine(params)` is the whole constructor**; Crank-Nicolson is the only built-in
  scheme. `theta=1.0` is backward Euler: first order in time and more diffusive, not
  "monotone" (monotonicity also needs the spatial operator's sign condition).
- The first-passage continuous-KI correction and the asymptotic boundaries are
  unconditional (their opt-outs are removed).
- **2D Heston/SLV damping:** event damping no longer follows the terminal switch
  (0.5's `use_rannacher=False` turned both off; set `event_damping_steps=0` too to keep
  that), and the 2D solvers read `terminal_damping_steps` as on/off (`> 0`).
- A budget-basis `GridConfig` resolves `day_count` to `None` (the day basis is not in
  force), so a resolved grid (`PDEParams.resolved_grid()`) can be fed back as an overlay.
- The grid "day" remains 1/252 of a solver year (`GridConfig.day_count`); it becomes the
  environment's business day in sub-project 3.

## [0.5.0] - 2026-09-28

Intraday pricing: `quantark.intraday` values a contract at a timezone-aware timestamp
instead of a date (spec: `docs/superpowers/specs/2026-09-15-intraday-pricing-design.md`),
and its certification lives in `quantark.modelvalidation`, not in the runtime
(spec: `docs/superpowers/specs/2026-09-18-intraday-modelvalidation-certification-design.md`,
reference revision: `docs/superpowers/specs/2026-09-18-intraday-deterministic-reference-revision.md`).
A certificate constrains the release, never the code. The autocallable engines agree on
the maturity close, on an observation at the valuation instant, on the knock-out-reset
snowball's contract rules and on a knock-out and knock-in at one instant (see Fixed).

The release also carries what `main` gained since 0.4.7: the simulated-path backtest,
futures bucket hedging in the replay engine, the opt-in QUAD V2 engines, PnL explain,
vol-model calibration and the Greeks facade (see "Also in this release").

### Added
- `quantark.intraday`: `value_intraday(request)` and `PricingSession.value_intraday`
  return an `IntradayValuationResult` with PV components, cashflows, a status per
  requested output and the provenance of every assumption. Three clocks: seconds-exact
  calendar time for carry and discounting, a versioned `VarianceProfile` (`desk`,
  `uniform`, `sessions_only`) mapped by `IntradayTimeMap` into `TradingClockVolSurface`
  for variance, and the contract's own day count for accruals. `TradingSessionCalendar`
  declares sessions, breaks, early closes and the payment time; a DST gap or fold is an
  error, never a guess.
- Fixings: confirmed `Fixing`s replay through the lifecycle trackers; a fixing that is due
  but not published is replaced by the latest spot and recorded as an `AssumedFixing`,
  and the flows it can change carry that dependency. `ObservationRecord.observation_timestamp`
  gives a timed observation (intraday only).
- Routes: QUAD V2 (snowball, Phoenix, knock-out-reset snowball), PDE (with an
  `under_resolved` diagnostic when the diffusion layer is not resolved), Monte Carlo,
  analytical digital (exact zero-variance limit) and analytical barrier/one-touch (gated
  by an exact admissibility test), behind an explicit capability matrix with fail-closed
  lookup (`docs/execution/intraday-capability-matrix.md`).
- Greeks: desk-bump Greeks on the frozen float-time twin, point Greeks per route
  (`undefined` at a payoff kink), local theta with a declared step and unit,
  `roll_context` and `roll_through_events`.
- Batch: `value_intraday_many`, prepared `spot_curve`s and provenance-preserving
  `aggregate_intraday`.
- Lifecycle: trackers declare coupon arrears and a settled terminal claim.
- `quantark.modelvalidation` schema 2: a quantity catalogue with per-quantity budgets,
  a `context` block with per-case overrides, semantic `expect:` cells, per-case random
  substreams, convergence axes with at least three levels, three-way verdicts on cells
  and aggregates, and a `contract` block an amendment may not change. Schema-1 studies,
  evidence and wire format are unchanged.
- A deterministic reference kind (`reference_kind = "deterministic"`): a typed record
  with a radius per quantity that declares itself `analytical` or a `calibrated_estimate`,
  consumed whole by the gate (`bounds.radius_budget_fraction`, no interval multiplier) and
  added linearly in the aggregate. `reference_qualification` names a stochastic arm that
  must agree with it case by case; an unqualified case is not decision-eligible, and the
  arm's sampling is frozen in the contract.
- Anchors of quantities that difference prices: a candidate may declare
  `anchor_noise_weights(case)`, the L1 weight of each quantity's price stencil, and off the
  banking machine such an anchor is compared at the prices' tolerance carried through that
  stencil (`modelvalidation.anchors.anchor_tolerance`). The intraday snowball candidates
  declare the runtime's desk bump and theta step. A one-second desk theta had moved 2.2e-8
  relative on x86_64 while the prices it differences agreed within 1e-9; the new tolerance is
  at most 4e-8 of a spot Greek's column scale and 2e-6 of desk theta's. Same-machine
  comparison stays exact.
- `python -m quantark.modelvalidation bank <run dir>`: validates, refuses quick runs,
  extracts anchors and never overwrites a banked directory.
- Study `snowball_intraday_daily_ki_bsm.yaml`: the daily-KI snowball on the intraday
  clock, 23 cases, QUAD V2 and PDE intraday routes against the engine-independent
  Gaussian-transition reference, qualified by paired RQMC. Banked 2026-09-19
  (`docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/2026-09-19`, digest
  `a0569e97687cc01c...`): `SnowballQuadEngineV2` ADMITTED, `SnowballPDESolver` REJECTED.

### Removed (breaking)
The import shims deprecated in 0.4.x are removed, as announced:
- `quantark.backtest.otc` and its twelve modules. Import from `quantark.backtest.replay`
  (`config`, `single`, `market`, `engine_factory`, `strategy_state`, `results`,
  `product_replay`, `dashboard`), `quantark.param.vol.surface_history` (was
  `otc.vol_history`) and `quantark.volcalibration.calibrate` (was `otc.vol_calibrators`).
- `quantark.volmodels.calibration`: use `quantark.volcalibration.calibrate` (the same module;
  a monkeypatch on its kernels reaches the calibrator as before).
- `VolModelCalibrationConfig` and `HESTON_PRESETS` are no longer re-exported from
  `quantark.backtest.replay` or `quantark.backtest.replay.config`: import them from
  `quantark.volcalibration.config`.

### Changed
- PDE delta and gamma are read off a local cubic through the four log-spot nodes around
  the spot (`BasePDESolver._calculate_delta_gamma`; a three-node grid keeps the quadratic).
  The nearest-node quadratic carried that node's constant curvature to the spot: an O(dx)
  gamma error whose sign flipped as refinement moved the nearest node across the spot.
  Every PDE engine's delta and gamma move; PV does not. The replay goldens moved by at most
  1.2e-4 of scale in delta and 3.6% in gamma. The four studies that anchor PDE Greeks were
  re-certified and banked 2026-09-28 with every decision unchanged (ADMITTED):
  `snowball-flat-bsm` (`ccba1c32d4055333...`), `phoenix-flat-bsm` (`8cd5584888009924...`),
  `ko-reset-flat-bsm` (`722b7726fadc4d41...`) and `snowball-localvol-1d`
  (`c0d62f9f4c0986a9...`); their predecessors are retired with `superseded_by`. On those
  cells the readout is accuracy-neutral (worst cells move by at most 0.004 of a
  0.5-contract bound).
- `quantark.intraday` returns what it computes. Every requested output carries its own
  status (`ok`, `undefined`, `failed`, `not_requested`) with diagnostics; the `unqualified`
  status, the `certify=` flags, `GreekValue.error_budget` and the packaged Gate C evidence
  (`quantark/intraday/evidence`) are removed. The PDE resolution verdict is `under_resolved`.
  The runtime reads no certificate and does not import `quantark.modelvalidation`.
- The Gate C evidence is archived under `docs/modelvalidation/legacy/intraday-gate-c/2026-09-18/`
  with an inventory of what each fixture's successor study is.
- A knock-out and a knock-in observed at the same instant: the knock-out wins, on every
  autocallable and every engine (desk ruling, 2026-09-21). The knock-in observed with it neither
  disables that knock-out under `disable_ko_after_ki` nor, on a knock-out-reset snowball,
  replaces the first schedule. The rule is for a discretely observed knock-in; a continuously
  monitored level breached at an observation was touched before it and keeps its precedence.
  One spot breaches both only when the knock-out level is at or below the knock-in level, so the
  usual contract is unchanged: 240 prices (three products, Monte Carlo, PDE, QUAD V1 in both
  event projections and QUAD V2, discrete and continuous knock-in, flag on and off) are
  byte-identical before and after. On a tie contract the engines used to disagree (spot 70,
  knock-out at 60 under a knock-in at 75, monthly observations): the snowball under
  `disable_ko_after_ki` was -22.4 on Monte Carlo, PDE and QUAD V1 against 14.0 on QUAD V2; the
  Phoenix 82.1 on Monte Carlo and both QUAD engines against 100.2 on PDE; the knock-out-reset
  snowball -21.6 against 14.0 on QUAD V2, with or without the flag. All four now agree with a
  semi-analytic reference (the snowball under the flag with two observations: 3.0803, where the
  knock-in winning gives -18.1045).
  Changed: `SnowballMCEngine` (both products) and `PhoenixMCEngine` (a knock-out at the knock-in
  instant is valid), `SnowballPDESolver` and `KOResetSnowballPDESolver` (the knock-out jump on
  the not-knocked-in surface goes after the knock-in jump where the knocked-in surface does not
  carry it, at interior observations and in the terminal condition), `SnowballQuadEngine`,
  `KOResetSnowballQuadEngine` and `PhoenixQuadEngine` (the same, in both event projections),
  and QUAD V2's Phoenix, which let the knock-in win under the flag on purpose. `PhoenixPDESolver`
  already let the knock-out win. The event-statistics sweeps of the PDE and QUAD V1 engines are
  not reordered, and the two-dimensional vol-model solvers were not examined.

### Fixed
- The maturity close under `BEFORE`, where every remaining event is decided on the known
  spot. The QUAD V2 intraday route raised there; the Monte Carlo route priced a claim that
  knocks in or out at that instant as if it had survived (12 where the lifecycle pays -30 at
  spot 70), and the PDE route paid a knock-out decided there as a survival (the rebate, not
  the knock-out coupon; its desk theta one hour earlier was wrong by 7 per hour). All three
  routes now resolve the decided claim through the runtime's own event resolution.
- The same defect at its source: the zero-maturity shortcut of `SnowballMCEngine`,
  `SnowballPDESolver` and `SnowballQuadEngine` returned the terminal payoff of the carried
  knock-in state and never applied the observations still pending at that instant, although
  each engine decides such observations correctly whenever more of the claim lies ahead.
  They now share `SnowballOption.decide_observations_at_valuation`. Day-level dated products
  cannot reach this state (valuation on the exercise date is refused); a time-based product
  at zero maturity can.
- The same defect in the other two autocallables. At zero maturity `PhoenixMCEngine`,
  `PhoenixPDESolver` and `PhoenixQuadEngine` returned the redemption alone: the last coupon,
  the memory arrears it releases, a knock-out and a knock-in at the last observation were
  dropped (Monte Carlo and QUAD V1 also dropped a knock-in carried from an earlier day). The
  knock-out-reset snowball's shortcut in `SnowballMCEngine`, `KOResetSnowballPDESolver` and
  `KOResetSnowballQuadEngine` ignored the pending knock-out of the schedule in force (pre-KI
  before a knock-in, post-KI after one), and QUAD V1 the carried knock-in. They now call
  `PhoenixOption.decide_observations_at_valuation` and
  `KnockOutResetSnowballOption.decide_observations_at_valuation`. The Monte Carlo and PDE
  intraday routes resolve the maturity close for all three autocallables; the knock-out-reset
  snowball, which the runtime cannot replay, is decided from its float-time twin, and its QUAD
  V2 route no longer raises there.
- The knock-out-reset snowball's two contract rules, on every engine: a knock-in REPLACES the
  first knock-out schedule by the second and stays to the end, and the second schedule only
  becomes effective through a knock-in DURING the first, so a contract not knocked in when the
  first schedule ends has matured there (`KnockOutResetSnowballOption.first_schedule_ended`,
  `require_alive`). `SnowballMCEngine` kept testing discrete knock-in records after the first
  schedule had ended (3.41 against 5.94 from PDE and QUAD on a contract with monthly knock-in
  records for a year and a six-month first schedule), read a spot-dependent rebate off the end
  of the time grid instead of the end of the first schedule, and ignored a knock-in carried
  into the valuation. `KOResetSnowballQuadEngine` always priced the not-knocked-in state and
  refused a contract whose first schedule had ended; it now reads the knocked-in surface, on
  a lattice pinned to the second schedule's level. Every engine, and the intraday twin, priced
  or refused the wrong way round past the first schedule: a knocked-in contract is now valued
  there on all of them, and one not knocked in is refused as matured instead of being priced
  with the first schedule extended to the final maturity.
- `SnowballMCEngine`, knock-out-reset contracts: the second schedule is in force from the
  observation that knocks the contract in, that observation included, as in the PDE and QUAD
  engines (and in this engine's own REBASED branch). Monte Carlo counted it strictly after the
  knock-in, which only shows when the second schedule's level is at or below the knock-in
  level: 4.60 against 4.75 there, now 4.74 +/- 0.01. The usual contract is unchanged.
- `PhoenixQuadEngine` never read `CouponBarrierConfig.initial_coupon_arrears`: memory coupons
  owed at the pricing date were worth nothing. They are released by the first period that pays,
  as the Monte Carlo and PDE engines release them.
- `AutocallableLifecycleTracker.settle_maturity_if_due` raised `TypeError` for every Phoenix
  alive at maturity: it called `get_payoff` with the snowball's signature. This reached the
  portfolio lifecycle manager and the backtest replay, not only intraday valuation.
- `PhoenixPDESolver`: a knock-out decided at the valuation instant (any instant, not only
  the last) left out the memory coupons in arrears, which Monte Carlo, QUAD V2 and the
  lifecycle tracker release with that period's coupon (101 against 106 with five owed).
- An observation at the valuation instant with more of the claim ahead. `PhoenixQuadEngine` and
  `KOResetSnowballQuadEngine` skipped it (their time grid drops time zero): no knock-out, no
  coupon, no missed-coupon memory and no knock-in from it, so a Phoenix observed now at 104 was
  94.54 against 101 on PDE and QUAD V2, and one at 90 exactly one coupon short.
  `SnowballMCEngine` refused a knock-out-reset contract observed now (`dt_array must have
  strictly positive entries`, in pricing and in the event statistics): its grid kept time zero as
  a simulation node where the plain snowball's grid reads such a record off the known spot. All
  three now decide the observation on the known spot, and each engine satisfies the identities
  that need no second engine: a coupon observed now raises the price by exactly the coupon, a
  memory coupon missed now equals the contract carrying it as arrears, and a knock-in observed
  now equals the contract carrying the knock-in (to 1e-8, on all four engines).
- A Phoenix coupon pays principal x rate x the period's own fraction. The daily lifecycle
  tracker passed no fraction to `get_coupon_payoff` and got a silent 1.0, so a monthly
  product paid about 12x more through the tracker than through any engine. The quotation
  is now declared, `AccrualConfig.is_annualized_coupon` (default annualized, as every engine
  already priced: engine PVs are unchanged), and `get_coupon_payoff` resolves the contract's
  own fraction or fails closed. The tracker also kept missed memory coupons only as a count
  and returned before the coupon loop on a knock-out; both now match the engines.

### Also in this release (merged to `main` since 0.4.7)
- `quantark.backtest.simulation`: simulated-path ensemble backtest for autocallable books.
  `MarketPath` batches from a stationary block bootstrap of the joint spot/vol/carry
  history, GBM or designed stress paths; the CFFEX IM chain read off each path's carry
  curve; a vectorised lifecycle and futures ledger; exact, spot-ladder and PDE
  life-surface repricing providers with in-memory and on-disk state caches; a sampling
  accuracy gate; spawn-pool path batching; a conformance oracle against the replay
  engine; distributions with expected shortfall, paired comparisons on matched paths and
  run persistence. Worked study: `example/snowball_simulated_paths`.
- Replay engine, futures carry buckets: immutable carry scenarios, signed bucket
  aggregation, joint delta and parallel carry hedge strategies, a multi-leg futures ledger
  with atomic execution, carry audits by independent repricing, stress of unquoted carry,
  and an opt-in term-structure dividend source.
- QUAD: opt-in QUAD V2 Gaussian autocallable engines (`quantark.asset.equity.engine.quad.v2`:
  `SnowballQuadEngineV2`, `PhoenixQuadEngineV2`, `KOResetSnowballQuadEngineV2`,
  `AutocallableQuadEngineV2`). `QuadParams.readout` selects `"legacy_linear"` (default) or
  `"transition"`, which evaluates the final transition at the spot instead of interpolating
  between nodes (no delta staircase); `QuadParams.align_cell_stretch` (opt-in, `None` by
  default) puts every barrier on a node at once. Defaults leave every price unchanged.
- `quantark.pnlexplain`: waterfall and Taylor attribution of PV changes, lifecycle
  transitions, bucketed factor coordinates, trading-clock theta, and portfolio, position
  and quoted-leg explain with a replay recorder.
- `quantark.volcalibration`: market quotes to IV surface to LV/Heston/SLV calibration,
  with quote normalization, admission checks, a run store and a CLI
  (`python -m quantark.volcalibration`).
- Risk measures: the `GreeksCalculator` facade over a `GreekDef` registry, higher-order
  Greeks (speed, zomma, dividend volga, analytical closed forms) and a dual-clock theta
  suite (charm, color, vega/gamma theta, 1D/1TD, exact mode).
- Changed: scalar rho parallel-shifts the rate curve instead of replacing it with
  `FlatRateCurve(r(T) + bump)`, which measured a curve reshaping on any term structure
  (98x the parallel-shift rho on a 1%/3%/5% curve); flat curves are bitwise unchanged. On
  a trading clock the vol bump unit is one point of the trading-quoted sigma.


## [0.4.7] - 2026-08-25

Settlement-date-aware payoff discounting across the equity option stack
(spec: `docs/superpowers/specs/2026-07-29-equity-option-settlement-date-support-design.md`,
plan: `docs/plans/2026-07-29-equity-option-settlement-date-support.md`).
Determination and payment are now two separate clocks: underlying dynamics,
monitoring and optimal exercise stop at determination; every cashflow
discounts at its own payment time with curve-exact discount factors.

Alongside it, autocallable batch pricing gets its performance pass — the PDE
solve loop no longer re-serializes the product on every time step, and the
QUAD event-stats recursion gains a fused KI walk plus an opt-in forward
transition-density mode (`docs/autocall-engine-perf/`).

### Added
- `quantark.asset.equity.settlement`: shared settlement timing kernel —
  `SettlementConvention` (business-day / calendar-day / year-fraction lag
  with business-day adjustment and calendar), `SettlementRequest`,
  `ResolvedPaymentTiming` (determination/payment times + curve-exact
  `determination_df`/`payment_df`/`delay_df`), `SettlementResolver`
  (`resolve_contingent` / `resolve_pending`). No fabricated dates: a
  date-based convention on a time-only determination raises.
- Product contract: `BaseEquityOption.settlement_convention`; the existing
  terminal `settlement_date` is now honored by engines as the
  terminal-payment override; per-record `ObservationRecord.settlement_date` /
  `settlement_time` drive event-cashflow payment (KO redemptions, coupons,
  hit-paid rebates) across analytical, MC, PDE and QUAD engines for the
  BSM / Local Vol / Heston / SLV families.
- American exercise obstacle reflects delayed cash receipt in PDE and LSMC
  (node-specific `delay_df`); analytical American approximations raise a
  capability error for settlement forms they cannot represent.
- Engines declare settlement/timing capabilities and raise precise
  capability errors before numerical work rather than silently pricing at
  determination.
- Lifecycle: pending-cashflow ledger
  (`quantark.asset.equity.lifecycle.cashflows`) carries
  determined-but-unpaid receivables at `amount * DF(valuation, payment)`
  through payment; `price(..., lifecycle_state=...)` propagates through
  engines, positions, portfolio valuation, Greeks and the consolidated
  replay backtest (settlement decomposition; KO termination waits for
  terminal cash to land).
- Payment-aware event statistics: determination/payment time arrays with
  undiscounted and discounted expected cashflows that reconcile to engine
  PV.
- QUAD autocallable engines: opt-in `QuadParams.event_stats_mode="forward_density"`
  computes the event distribution from a forward transition-density march
  (2–3 surfaces instead of one indicator row per KO observation). `npv` is
  unchanged (always the backward value solve); distribution fields differ
  within the banked validation tolerances
  (`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`).
  KO-reset ignores the flag. Default remains `"stacked"`.

### Changed
- Snowball/Phoenix PDE: the product cache token is now memoized per solve
  (`SnowballPDESolver._product_token_memo`, cleared at every solve entry).
  The boundary-condition path previously re-serialized the whole product —
  every KO/KI schedule record — once per time step, >90% of a snowball
  solve; bit-identical output, ~7× faster per solve
  (`docs/autocall-engine-perf/FINDINGS-2026-08-24.md`).
- QUAD stacked event stats: the KI-probability recursion now rides the main
  stacked recursion as fused surfaces (bit-identical output, one fewer full
  time loop per `price_with_events`); identically-zero indicator rows are
  skipped in the diffusion/bridge steps and the row-independent bridge
  kernel stack is cached per engine (64-entry bound). Bit-identical output.
- QUAD `price_with_events` now honors `streams` pruning [§11.1], mirroring
  the PDE contract: KI-probability recursion and Phoenix coupon rows are
  skipped when no requested stream reads them. Callers that pass `streams`
  (the execution framework's greek-bump runner does) get `0`/empty for
  pruned distribution fields where QUAD previously returned computed values;
  `ko_probability`/`survival`/`pv` and all requested fields are unchanged.
  `streams=None` keeps the old full computation.
- **The flat dividend/carry sanity bound moves from 20% to 100%**
  (`ContinuousDividendYield`, the BSM process gate, and the market-data
  point model), matching `TermStructureDividendYield`'s own magnitude
  bound. OTC borrow/lending carry (融券率) legitimately exceeds 20% on
  some trades; the 100% ceiling still rejects unit errors (entering
  `26.89` for `0.2689`). Values between 20% and 100% now price instead of
  raising `ValidationError`.

### Fixed
- Repository tooling only — not shipped in the wheel, which contains
  `quantark/` alone. The `mo_volmodels` dashboard decided whether a results
  aggregate was stale by comparing two ISO-8601 stamps as strings, which
  orders correctly only when both carry the same UTC offset. Read from a
  zone other than the one that wrote the cells, an aggregate *newer* than
  the cells it summarises was reported stale. It now compares instants.
- Repository tooling only. The same dashboard read git commit dates with
  `datetime.fromisoformat`, which rejects an RFC 3339 `Z` suffix on Python
  3.10 (3.11 was the release that learned it). `git log --format=%cI`
  renders a UTC commit date as `...Z` on some git versions, so reading a
  repository committed under UTC raised `Invalid isoformat string` on the
  oldest supported Python. All three readers now share one parser that
  accepts both spellings.

### Compatibility
- Zero-lag identity: with no settlement terms supplied, payment equals
  determination (`delay_df == 1`) and all prices, Greeks, event statistics
  and exercise boundaries are unchanged — verified by the existing suite
  plus the dedicated settlement test files.

### Also in this release
- Constructor normalization forwarding the shared lifecycle fields across
  the concrete equity option products (wip/equity-option-contract).
- Heston temporal regularization in the ADI core
  (wip/heston-temporal-regularization).
- PDE regression coverage ports, per-module reference docs, example suites
  (FX vol models, snowball backtest terms, bucketed-greeks scenarios), and
  modelvalidation certification archival.

## [0.4.6] - 2026-08-20

0.4.5's test suite passed on the ARM64 machine its goldens were frozen on and
failed 28 tests on x86_64 Linux. Chasing that down found one genuine pricing
defect and a set of comparisons that were architecture-dependent by accident.
The published 0.4.5 package is unaffected except as noted below.

### Fixed
- **Parallel-batch pricing of continuously monitored knock-in under Local Vol /
  Heston / SLV Monte Carlo.** The per-step variance the Brownian-bridge crossing
  estimate needs was held in a single engine attribute, written while simulating
  and read while computing payoffs. The dask path (`use_dask=True`) submits every
  batch through `delayed`/`compute` on the threaded scheduler, so batches sat
  between their own generation and their payoffs concurrently on one engine and
  overwrote each other — 20,000 paths over 3 batches is 6667/6667/6666. Now
  scoped per thread, which is where a batch begins and ends. A shape guard made
  this raise rather than price one batch against another's variance, so no wrong
  number could have been returned; serial pricing was never affected.
- `quantark.modelvalidation.anchors`: comparing a banked certificate off the
  banking machine now floors the tolerance at the suite's cross-architecture
  standard (`DEFAULT_REL_TOL` 1e-12 → 1e-9, `DEFAULT_ABS_TOL` 1e-15 → 1e-12).
  Measured across 326 anchors, x86_64-vs-ARM64 drift spans 1e-14 to 2.2e-11
  relative (median 2.0e-12, both signs, no outlier of a different order): last-ULP
  differences in FMA contraction, libm transcendentals and BLAS kernels, amplified
  by an autocallable PDE/QUAD solve marching hundreds of steps through barrier
  events. The bound keeps ~44x margin over that and stays six orders under any
  real numerics change. Applied as a floor on each banked file's own tolerances,
  so already-banked evidence is corrected without its anchored *values* being
  touched. Same-machine comparison remains exact.

### Changed (tests)
- The replay golden gate compares floats against each column's scale rather than
  bitwise. These frames carry bump derivatives, so the rounding noise is set by
  the magnitude of the price being differenced and is near-constant in absolute
  terms down a column: gamma drifts 4.6e-10 against a column max of 43, and the
  same quantity in cash drifts 3.8e-8 against 4,301 — one ratio, ~1e-11. Column
  names and order, row count, and integer, string and date columns still compare
  exactly. New tests pin the tolerance itself: it admits the measured drift,
  rejects 1e-6 of column scale, and every real golden frame is checked against a
  simulated drift locally, because bitwise-identical frames short-circuit before
  any tolerance is consulted and made earlier calibration bugs invisible here.
- Two cross-architecture anchor tests hardcoded the fingerprint
  `{"machine": "x86_64", "system": "Linux"}` — which is the CI machine, so there
  they took the same-machine *exact* branch, the opposite of what they verify.
  They now derive a fingerprint that cannot be the current machine.
- `test_event_stats_pruning` pinned a PV of ~9.9e5 with a `1e-6` absolute
  epsilon, i.e. 1e-12 relative — tighter than the measured drift, passing on
  margin rather than by design. Now on the documented cross-arch tolerance.
- A settlement fixture placed a strike exactly on the parity-implied forward.
  Stage 10 splits OTM puts from calls with `strike < forward` against an
  OLS-regressed forward, so that strike changed side with the last ULP of the
  regression. The grid now keeps every strike clear of the forward.

## [0.4.5] - 2026-08-19

Two programs land together: **model-release certification** — the first
module that decides whether an engine is fit to ship, and the numerics
fixes that decision surfaced — and the **backtest replay consolidation**.

### Model-release certification

#### Added
- `quantark.modelvalidation`: engine-release certification. A study YAML
  declares the product, the market environment, the candidate engines and
  the economic scale; the runner prices every (candidate, case) cell against
  a paired-RQMC benchmark, applies per-cell and aggregate-bias gates, and
  emits a signed certificate, a Markdown/HTML report, and an *anchor file*.
  Anchors are the cheap residue of an expensive run — the deterministic
  engines' own outputs at the certified configurations — so CI can re-check
  in seconds that a released engine still produces what its certificate
  claims (exact on the banking machine, tolerance-based elsewhere).
  `amend` extends a certification's scope while carrying unaffected cells
  forward on identity hashes; scope may grow but never shrink.
  Procedure: `docs/modelvalidation/RELEASE_PROCEDURE.md`.
- Banked flat-BSM certifications for snowball (23 cases), phoenix (14) and
  KO-reset snowball (14), PDE and QUAD both ADMITTED, under
  `docs/modelvalidation/certificates/`, covering the full product feature
  surface: discrete / European / stepping knock-in, step-down and parachute
  knock-out, reverse, airbag, protection types, participation, call rebate,
  `disable_ko_after_ki`, and EXPIRY coupons.

#### Fixed
- **Continuous knock-in was monitored at the time-step width** by the
  local-vol, Heston and SLV solvers, which were gated off the FIRST_PASSAGE
  correction. The correction is barrier-local, so it never needed globally
  constant coefficients — only the dynamics at the barrier. Each solver now
  reports its own: `sigma_loc(barrier, t)` for local vol, one
  `(mu, sigma^2)` per variance column for Heston, and `L(barrier,t)^2 * v`
  for SLV. On a flat surface the local-vol solver had been bitwise identical
  to the *uncorrected* flat-BSM solver (+0.099% of PV). Applies to the
  snowball and phoenix families alike; the phoenix 2-D solvers had never
  built the state at all.
- **The Monte Carlo continuous-KI Brownian bridge used the implied vol at
  the strike** as the variance of every path and every step. The vol-model
  engines now supply the log-variance their own scheme accumulated. Against
  the closed-form first-passage probability the error falls from -10.0 / -8.5
  / -9.2 standard errors (Heston / local vol / SLV) to inside 4.
- The first-passage correction estimated its slope from a node that could sit
  a fraction of a cell above the barrier, where the knock-in mask's own step
  — not the slope — dominates. Sampling is now floored at half a cell.
- `PhoenixOption` never built its knock-in observation schedule, so a
  discretely monitored phoenix raised instead of pricing.
- `disable_ko_after_ki` was ignored by the snowball and phoenix PDE solvers,
  which wrote the knock-out payoff to the knocked-in surface as well.
- Phoenix `coupon_pay_type=EXPIRY` coupons roll up and are paid at
  termination — the knock-out settlement when it knocks out, maturity
  otherwise — and are never forfeited. All three engine families agree;
  PDE and QUAD gained a termination-value surface to price it.

#### Changed
- Continuous knock-in prices move in the local-vol, Heston and SLV engines
  (PDE and Monte Carlo). Discretely monitored, European and no-knock-in
  products are bitwise unchanged, verified across 54 engine/product/
  monitoring combinations; the sole exception is the opt-in
  `KnockInMonitoringMode.BGK_APPROXIMATION`, which replaces a discrete
  schedule with continuous monitoring by design.

### Backtest replay consolidation

(spec: `docs/superpowers/specs/2026-07-30-backtest-replay-consolidation-design.md`).

#### Added
- `quantark.backtest.replay`: canonical home of the product-replay backtest —
  ONE multi-product daily loop (`ReplayBacktestEngine`/`ReplayBacktestConfig`/
  `ReplayProduct`/`ReplayBacktestResults`); the single-product
  `AutocallableBacktestEngine` is now a book-of-one wrapper with byte-identical
  output (golden-gated). Book runs gain per-day vol-model calibration.
- `quantark.backtest.futures_ledger`: shared `FuturesHedgePosition` +
  `FuturesRollPolicy`.
- `quantark.backtest.metrics.CorePerformanceMetrics`: cross-asset metrics over
  the `BaseBacktestResults` protocol; replay results expose `.metrics` and the
  full protocol; equity `PerformanceMetrics` now extends the core.
- `quantark.backtest.replay.schema`: typed row schemas — the single source of
  truth for every record column.
- Pending-settlement KO termination: `terminate_on_lifecycle_end=True`
  (default) ends a replay when terminal cash lands (observation vs settlement
  are separate moments; discounted receivable carried in portfolio value);
  summaries gain `termination_reason` / `days_replayed` / `days_in_contract`.
- `AutocallableEngineConfig.event_stats_fallback` (`"none"` default): the
  silent 5000-path MC event-stats fallback is now opt-in, logged, and recorded
  per row in the new `event_stats_engine` column.
- `BumpConfig.gamma_spot_bump`: engine-side asymmetric gamma bumping;
  replay-side manual bumping removed (greeks are always
  `engine.calculate_greeks`, failures raise — no silent zero-delta days).
- `AutocallableDeltaHedgeStrategy` joined the `BaseStrategy` hierarchy
  (`quantark.backtest.strategy`).

#### Changed
- Relocations (old paths remain as `DeprecationWarning` shims until 0.5.0):
  `quantark.backtest.otc.*` -> `quantark.backtest.replay.*`;
  `otc.vol_calibrators` -> `quantark.volmodels.calibration`;
  `otc.vol_history` -> `quantark.param.vol.surface_history`;
  `_atomic_write_json` -> `quantark.util.io.atomic_write_json`.
- Replay books with `vol_model != "bsm"` reject non-Snowball products at
  config construction (the vol-model engine factory is snowball-only).
- **Heston/Heston-SLV Snowball and Phoenix 2D PDE solvers default
  `v0_boundary="degenerate_pde"`** (was the ADI core's `"neumann"`), and accept
  it as a constructor argument. Smile-calibrated Heston sits near the Feller
  boundary, where the Neumann treatment of the `v = 0` edge mis-prices by
  −0.540% of notional against a QE-M MC reference while the degenerate
  treatment holds +0.156%. `"neumann"` remains selectable for cross-checks.
  **Prices change** for these four solvers at default settings.
- **Heston/Heston-SLV Snowball concentrated PDE grids default to a power-graded
  variance axis** (`v_grid_power=2.5`) to resolve the near-zero layer around the
  Feller boundary. `v_grid_power=0.0` retains the legacy concentrated mesh for
  explicit cross-checks; uniform grids remain ungraded. **Snowball Heston/SLV
  prices and finite-difference Greeks change** at default settings.
- **The `mo_frozen` Heston preset enforces the Feller condition**
  (`enforce_feller=True`; the soft `regularize_feller` penalty is retained).
  Deliberate divergence from the `mo_volmodels` diagnostics scripts, which keep
  the soft-only policy. Feasibility is bought with smile accuracy — measured
  across 762 CFFEX MO settlement surfaces: median 8.4 bp, p90 29.3 bp, max
  217 bp of IV. On 6.6% of dates the constraint is met by driving vol-of-vol to
  its lower bound, which degenerates Heston to deterministic variance; consumers
  should screen on the new `feller_ratio`. Heston and Heston-SLV calibration
  cache entries are invalidated automatically (the fingerprint embeds the
  preset); `localvol` entries are unaffected.

#### Fixed
- `BaseEngine.calculate_greeks` and Gate G2's deterministic delta helper bypassed
  `create_bump_context`, so Heston/SLV central spot bumps rebuilt their S grids even
  though the solvers already provided a frozen-layout context. Base, up, down, and
  separate gamma reprices now all use the resolved base-market context.
- `Heston2DAutocallableSessionAdapter` now preserves the Snowball solver's
  `v_grid_power`; an explicit grading or legacy opt-out no longer disappears when
  the execution session clones the engine. Gate metadata records the selected
  value and Stage 12 forwards it into the production engine options.
- `calibrate_heston(enforce_feller=True)` failed closed on any problem where the
  constraint was active at the optimum and `ftol` was loose. SLSQP satisfies
  constraints only to about its own accuracy tolerance, but the feasibility
  buffer was a constant `1e-8` — measured slack is ~1e-9 at `ftol=1e-8` but
  ~1.6e-7 at `ftol=1e-6`, so the strict post-check rejected every fit. The
  buffer now scales as `max(1e-8, 10 * ftol)`; the post-check is unchanged and
  still fails closed.
- `VolModelCalibrator` Heston records carry `feller_ratio` (`2*kappa*theta /
  sigma**2`) alongside `feller_margin`. The margin is a difference and so is not
  comparable across dates; the ratio is.
- `Heston2DAutocallableSessionAdapter._clone_engine` rebuilds the engine from an
  explicit kwargs list and dropped `v0_boundary`, so a session or prepared-adapter
  run silently priced with the default boundary while the direct call honoured the
  constructor. Any 2D autocallable constructor argument omitted from that list
  fails this way — silently, with no error.

## [0.4.0] - 2026-07-27

Clean rewrite of the PDE grid construction and event application layer
(spec: `docs/superpowers/specs/2026-07-27-pde-grid-redesign-design.md`).

### Added
- `quantark.asset.equity.engine.pde.grid`: declarative grid layer —
  `GridRequest` (frozen geometry), `GridConfig` + accuracy profiles
  (`fast`/`standard`/`high`), ONE time builder (event-aligned, extras-scaled
  cap, damping schedules), ONE spatial builder (multi-point sinh, pinned
  auto-bounds, no barrier snapping), four-stage `EventSchedule`
  (terminal/interior/continuous/valuation-readout, pure block transforms),
  engine-owned `GridBinder` (LRU, `bind_shared` for same-underlying books,
  `rebind_time` for calendar rolls) and `GridLayerMixin` for
  BaseEngine-derived adopters.
- `PDEParams(accuracy=..., grid=GridConfig(...))` tiered front door.
- Frozen-`Layout` bump contexts: spot/vol/rate/div bumps reuse the base
  layout by object identity; calendar rolls rebind time on the same spatial
  object.
- `HestonSLVADICore(x_nodes=...)`: 2D solvers take their S-axis from the
  shared spatial builder.
- Tier-2 anchor certification (`test/pde_grid/test_anchor_certification.py`):
  MC/QUAD/closed-form/smoothness anchors gate every accuracy profile.

### Removed (breaking)
- Legacy grid modules `time_grid.py`, `spatial_grid.py`,
  `event_projection.py` (projection math moved verbatim to `grid/events.py`).
- `PDEParams` knobs: `auto_grid`, `time_grid_type`, `grade_exponent`,
  `event_min_steps_per_interval`, `log_dx_target`,
  `include_spot_in_critical_points`, `frozen_critical_points`,
  `barrier_refine_log_width`, `barrier_refine_levels`,
  `barrier_domain_expand`, `adaptive_grid`, `event_steps_per_day`,
  `max_time_steps`, `max_grid_size` (the last three had no remaining
  consumers — `GridConfig.steps_per_day` / `max_steps` / `max_points` own
  those roles).
- The class-level shared grid cache (`clear_grid_cache` is now a no-op).

### Changed
- All PDE prices reprice slightly (re-certified against MC/QUAD/closed-form
  anchors, goldens re-frozen): the layer builds concentration-based grids
  without barrier snapping — cell-average event projection is placement-
  independent — and event-aligned time grids emit one exact dt per interval
  (operator caches are arithmetic-neutral: cache-on == cache-off bitwise).
- Grid-layer solvers REJECT non-default `grid_size`/`time_steps`/`s_min`/
  `s_max` with a `ValidationError` naming the `accuracy`/`GridConfig`
  replacement (previously the values would have been silently inert); the
  knobs remain live inputs for the standalone vol-model solvers only.
- `make_pde_params` profiles (`fast`/`accurate`/`barrier_sensitive`/
  `reverse_sensitive`) now emit `accuracy`/`grid=GridConfig(...)` instead of
  the legacy knobs (which also fixes a crash: the product-hint path injected
  deleted `barrier_refine_*` keys).
- Bump contexts (`create_bump_context`) clone via `deepcopy`, preserving
  full constructor state (Heston `model_params`, SLV leverage surfaces, ADI
  dimensions); `PDEEngine.create_bump_context` carries the frozen solver
  instance. Frozen layouts fail closed when reused across different hard
  (absorbing-barrier) bounds, and calendar-roll rebinds coverage-validate.
- 2D Heston/SLV snowball/phoenix bump contexts freeze the ADI S-axis at the
  solve configuration (`n_x`, `num_std=8`) and reuse it by identity under
  market bumps (`_layer_x_nodes` routes through `resolve_bound_layout`);
  previously the S-axis was rebuilt per bump, mixing grid movement into 2D
  numerical greeks.
- The scheme knobs are LIVE on the grid layer: `use_rannacher`/
  `rannacher_steps` drive terminal damping and `rannacher_at_events`/
  `event_rannacher_steps` drive event damping (the binder derives the
  damping schedule from them; explicit `GridConfig` fields still win). At
  default knob values the schedule is unchanged.
- Prepared/session PDE pricing resolves solve state
  (`_prepare_for_request`) before fingerprinting grid geometry — BGK
  products no longer trip `DeterminismViolation` against their own
  preparation when an artifact cache is active.

### Fixed
- Autocallable PDE grids ignore out-of-reach critical-price markers (such as
  disabled 100× KO observations) for bounds and concentration while retaining
  the observation date and any coupon event. Dense critical sets now use
  bounded cumulative inversion of the Tavella-Randall monitor, removing the
  overflow-prone shooting path and its non-monotonic construction-time cliff.
- Snowball/Phoenix QUAD event handling now defaults to cell-average projection
  and phase-stable trapezoidal integration. Unreachable KO markers are
  suppressed without deleting their dates, and the legacy nodal/Simpson
  behavior remains available explicitly.
- QUAD can optionally refine nested odd grids until consecutive PV estimates
  satisfy configured absolute/relative tolerances. Fixed
  `grid_points=1001` remains a speed-oriented estimate rather than a universal
  convergence guarantee.

### Deferred (recorded follow-ups)
- Event-aligned 2D ADI time axis (uniform march retained).
- Standalone vol engines (`LocalVolPDESolver`, `HestonPDESolver`,
  `HestonSLVPDESolver`, LV/Heston/SLV barrier engines, `HestonDCNPDESolver`)
  keep bespoke kernels and their `grid_size`/`time_steps`/`s_min`/`s_max`
  inputs; `event_projection` + Rannacher scheme knobs survive for the live
  characterization paths.
- Stencil-delta accuracy on concentrated grids (~2e-3 absolute) —
  provisional 5e-3 anchor band.

## [0.3.0] - 2026-07-20

First release of the `quantark.execution` framework kernel and the
`quantark.execution.greeks` scenario layer. The `test_snowball_quad_flat_identity_golden`
failure predates this program and remains quarantined (reproduces on
unmodified main). The spec §20 controlled-host performance gates in
`docs/execution/README.md` are documented as release-preparation evidence
and were not re-measured on the release host.

### Added
- **`quantark.execution.greeks` — greek bumps as scenario cells** (spec
  2026-07-20): `TradeState` per-trade base, same-type `greek-bump/v1`
  transformer with real spot/vol/rate/div/time mutation attribution,
  `greek-value/v1` float runner, and assemblers mirroring
  `EquityPosition.get_trade_risk` / `GreeksCalculator.
  calculate_numerical_greeks` operation-for-operation (bitwise unit gate).
- **`PricingSession.run_scenario_plans`** — many `(base, specs)` plans
  packed through ONE bounded-window process pool (per-cell worker-spec
  payloads; the portfolio × bumps shape), with a per-plan error boundary
  under `collect_errors=True` (base-resolution/planning failures become
  aligned typed `PricingFailure`s while other plans execute).
- **`quantark.execution` — composable execution kernel** (framework contract
  v1; spec `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`).
  `PricingSession` wraps every exported MC/PDE engine without changing direct
  legacy behavior: adapter registry, resource budgets/leases, prepared-artifact
  and session draw caches, reproducibility manifests and immutable diagnostics.
  Capabilities: fixed-batch MC backends (serial/threads, bit-identical
  reductions in canonical order), adaptive RQMC session mode (bitwise vs
  direct across the 12 autocallable vol engines), PDE preparation artifacts
  (grids, step coefficients, factorization packs; one-solve PV/event/grid
  outputs across 18 prepared engines + FX LV), typed scenario execution with
  verified mutation footprints, spawn-safe `WorkerSpec` process workers, a
  Dask backend over the same plans, `price_many` grouping, and a
  complete-payload scenario validator. Published artifacts: generated
  [capability matrix](docs/execution/capability-matrix.md) (CI-fresh), policy
  guide, legacy-internals rationale, reproducibility JSON Schemas (validated
  against live payloads), and runnable migration examples. Adoption is
  inventory-driven and honest: `temporary_legacy` rows remain, and no
  universal framework support is claimed until the controlled-host gates pass.
- **DCN (Digital Coupon Note)**: product + observation-schedule generator,
  curve-aware MC engine with leg decomposition and event stats, two-surface
  PDE engine with MC↔PDE cross-validation gates, LocalVol/Heston/QE/SLV DCN
  MC engines sharing one payoff kernel, MLMC-style coupled timestep-ladder
  Heston pair, SVI/cleaned-quote calibration layers with no-arbitrage and
  repricing-residual reports, node-bump vega buckets and carry-invariant rate
  bumps.
- **Implied futures carry**: `IndexFuturesCurve` implied q(T), futures-tenor
  delta/rhoq buckets, delta-one futures rhoq, portfolio aggregation and hedge
  integration.
- **Engine term-structure upgrade**: MC, PDE, and quadrature autocallable
  engines consume full r/q/vol term structures with a cross-family term
  agreement gate.
- **Vol-model barrier exotics**: LV/Heston/SLV barrier MC pricers on a shared
  monitoring core, LV 1D barrier PDE, Heston/SLV 2D barrier ADI solvers.
- **Vol-model numerics program (phases 2–5)**: vectorized Lewis Heston
  calibration (~21x), unified Heston/SLV ADI core, opt-in Krylov and TR-BDF2
  Fokker–Planck marches, opt-in QE-M martingale correction, concentrated
  (x,v) grids, degenerate v=0 Feller boundary, LV Rannacher start-up
  (default on), opt-in QMC sampling for Heston/SLV/LV kernels.
- **PDE event-stats API** on Snowball/Phoenix vol solvers with event
  distributions surfaced through session outputs.

### Changed
- DCN MC engine performance: draw cache, batched threads, LV build hoist.
- The legacy autocallable Dask batch loop (Snowball vanilla/KO-reset,
  Phoenix) is consolidated into one shared reducer
  (`autocallable_dask_batch`) with byte-preserved behavior, gated by frozen
  bitwise goldens. The legacy `use_dask` route itself is unchanged and
  preserved (see `docs/execution/internals-and-legacy.md` for the §17.3
  removal preconditions).
- Dev extras now include `dask[distributed]` and `jsonschema`.

### Fixed
- `PhoenixMCEngine` silently disabled its Dask parallel path on modern dask
  (`from dask.compute import compute` no longer resolves); availability now
  matches `SnowballMCEngine`, with a regression test.
- Processes scenario backend: a parent whose `__main__` cannot be
  re-imported by spawn children (stdin/heredoc scripts record
  `__file__ = '<stdin>'`; deleted script files) now fails closed with a
  typed `CapabilityError` before any worker is spawned, instead of every
  child dying at bootstrap and surfacing as an opaque
  `BrokenProcessPool` with zero completed cells.
- Craig–Sneyd ADI corrector order restored (base `Y0` + implicit `-rU`),
  QE sampler sign bug, and the 2026-07 PDE/QUAD audit fixes (13 findings)
  are included via the programs above.

## [0.2.5] - 2026-07-03

### Fixed
- 2026-07 PDE/QUAD audit: 13 numerical findings fixed (discrete-monitoring
  boundary handling, Rannacher terminal off-by-one, BGK state resolution,
  smoothed KI for Phoenix/KO-reset, shared theta damping schedule) with
  solver-family consolidation and regression tests.

## [0.2.4] - 2026-07-02

### Added
- Opt-in BGK continuous-KI mode for Snowball and Phoenix PDE solvers
  (direction-aware shifted barrier, per-product flag).
- `EquityPosition.get_trade_risk` (product + total, frozen one-loop) and
  `AutocallableCashLeg.time_shift`; event streams selected by leg
  requirements.

### Fixed
- Portfolio vega/rho/dividend-rho aligned with the canonical greek
  convention; bump contexts freeze PDE critical points for grid steadiness.

## [0.2.3] - 2026-07-01

### Added
- Cross-engine-consistent `ki_ever` / `ki_survive` event-stat fields.

### Fixed
- Theta advances by business day; quadrature event-stats recursion uses
  smoothed barrier indicators.

## [0.2.2] - 2026-07-01

### Fixed
- PDE numerical Greeks: freeze the base spatial domain for finite-difference
  bump repricing so rho and dividend rho measure market sensitivity without
  contamination from auto-grid/domain movement under bumped rate, dividend,
  volatility, or time inputs.

## [0.2.1] - 2026-06-30

### Fixed
- `PhoenixPDESolver`: apply a KO observation scheduled exactly at maturity.
  The inherited grid builder stores the maturity KO in `_ko_terminal_record`
  (intentionally kept out of `_ko_observation_indices`), but the Phoenix
  `_solve` override looked the terminal KO up in `_ko_observation_indices`
  and therefore dropped it — mispricing products with a terminal KO date by
  several percent versus the quadrature and Monte Carlo engines. Terminal KO
  is now applied after the terminal coupon/KI jumps (matching
  `SnowballPDESolver._solve`), routed through `_apply_ko_jump_vector` so the
  same-date coupon-at-KO payoff is preserved.

## [0.1.2] - 2026-06-13

### Added
- Credit dual-measure framework: a recovery convention layer
  (`quantark.asset.credit.conventions`, `STANDARD_RECOVERY=0.40`) that
  separates the canonical shared-curve **hazard01** factor from the
  recovery-converted **CS01** used by products and SIMM. Curve shocks stay
  in hazard space; spread stresses convert through the recovery convention.
- Single-name CDS **roll-down / as-of pricing** via effective and maturity
  dates (seasoned and forward-start), with `schedule_asof` and a
  total-return coupon cash ledger threaded through the dynamic-scenario and
  backtest engines. SIMM buckets the remaining tenor. (Basket as-of is
  deferred.)

## [0.1.1] - 2026-06-11

### Added
- `SnowballQuadEngine`: explicit `ki_monitoring_mode` on `QuadParams`
  (`KnockInMonitoringMode`). `EXACT_DISCRETE` (default) prices every KI
  observation date exactly with adaptive spatial-grid refinement.
  `BGK_APPROXIMATION` is an opt-in performance mode that replaces a dense
  discrete KI schedule with continuous monitoring at a
  Broadie-Glasserman-Kou shifted barrier; the engine validates approximately
  regular spacing (median-band dispersion test), a constant resolved
  barrier, full-horizon coverage, stable volatility, and a minimum schedule
  density (`bgk_min_ki_observations`), raising `ValidationError` otherwise.
  Converted pricing matches the equivalent shifted-continuous product
  exactly (grid-aligned to the shifted barrier) while the valuation-time KI
  state keeps contractual discrete semantics. A first-order residual bias
  remains (a few bp of PV at daily spacing, growing with observation
  spacing and drift).

### Fixed
- `SnowballQuadEngine`: dense discrete KI schedules now retain their explicit
  observation dates instead of being delegated to continuous monitoring. The
  engine adaptively refines its internal spatial grid to resolve short
  diffusion intervals (accuracy-oriented default of 2.5 cells per interval
  diffusion stddev; lower to 1.25 or opt into BGK for speed), avoiding the
  material fair-KO-rate bias caused by the previous continuous-monitoring
  approximation.

## [0.1.0] - 2026-06-11

### Added
- First public release.
- Equity derivatives: European/American/Asian vanilla options, barrier,
  one-touch, digital, sharkfin, and autocallable products (snowball,
  phoenix, KO-reset snowball, range accrual) with analytical, Monte
  Carlo, PDE, quadrature, and tree engines.
- Fixed income: fixed bonds, FRNs, bond options, bond forwards/futures,
  convertible bonds, interest rate swaps.
- Market data layer (`quantark.param`, `quantark.priceenv`), Greeks
  calculators, portfolio VaR (parametric/historical/Monte Carlo),
  ISDA SIMM v2.6, stress testing, multi-day scenario simulation, and a
  hedging backtest framework.
- Legacy flat-import compatibility shim (`asset`, `util`, …) with
  `DeprecationWarning`; slated for removal in 1.0.
