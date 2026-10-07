# Theta at an Observation Boundary — Design Spec

**Date:** 2026-10-07. **Status:** design approved in session (three rulings below); this written
spec awaits review. **Origin:** handoff item 6 of
`docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md` §12 ("`time_shift` on a
schedule it empties").

## 1. Problem

Theta reprices the contract one bump later: `t0` is today's as-of date, `t1` the bumped date
(one calendar or business day, so Friday to Monday is one bump). Every theta path moves the
product with `product.time_shift(time_bump, bumped_date, env)` and reprices it. Three defects
sit in that step.

1. **The observation on `t1` is skipped.** `ObservationSchedule.time_shift`
   (`observation_schedule.py:337-375`) drops a numeric record when `t - bump <= 0` and a dated
   record when `observation_date <= bumped_date`, so a record exactly on `t1` disappears; every
   override inherits the rule (`BarrierConfig._shift_side`, `snowball_config.py:151` keeps legacy
   dates only when `t - bump > 0`; `AutocallableCashLeg.time_shift`,
   `cashleg/autocallable_leg.py:207`, keeps only `obs - dt > Tolerance.ZERO`). With daily
   monitoring `t1` is an observation date on every bump. Measured: a daily accumulator (strike
   95, knock-out 105, 10,000 shares a day, spot 100, first observation at `t1`) has theta
   +55,402 on a value of −154,177 — one observation's worth.
2. **The outcome decided at `t0` is lost.** When `t0` is an observation date the engines decide
   its record on today's spot (`at_valuation_instant`, `decide_observations_at_valuation`, the PDE
   `_valuation_instant_records`). The bumped product drops that record, as it should, but nothing
   carries what it decided: an accumulator's shares (`time_shift` never extends
   `past_observations`), a knock-in triggered at `t0` (numerical theta prices the bumped product
   with no lifecycle state), a range accrual's fixing (time-based records are never shifted).
3. **A schedule the bump empties zeroes theta.** When every record is past but maturity is not,
   `ObservationSchedule.time_shift` returns `None`, the product's `time_shift` returns `True`
   before it shortens maturity, and every caller reads `True` as theta 0
   (`riskmeasures/greeks/numerical.py:254-259` and `:595-600`, `execution/greeks.py:175-177`,
   `portfolio/equity/position.py:377-379`, `report/autocallable_risk_report.py:1051`, `:1572`).
   That value is invented.

## 2. Rulings (user, 2026-10-07)

| # | Ruling |
|---|---|
| R1 | Theta moves one day forward. `t0` is gone: its record is dropped even when `t0` is an observation date. A record on `t1` is kept only if `t1` is an observation date of the contract, and it is observed at the new as-of instant (time 0), never skipped. Records strictly between `t0` and `t1` are past. |
| R2 | Only the theta paths keep the `t1` record. Lifecycle aging (`lifecycle/autocallable.py`, replay, backtest repricing, the dashboard) has already observed and booked `t1`'s events, so it keeps dropping the record; its behaviour is byte-identical. |
| R3 | A bump past the last observation (maturity later) prices the remaining contract with no monitoring left; the bumped product is always aged. Once observation ends the payoff is fixed by what was observed, so theta is carry: discounting, plus the time value of whatever terminal payoff remains (a knock-out never hit is a European). |
| R4 | The outcome decided at `t0` is carried into the bumped product's state, decided on the spot exactly as the engines decide it at `t0`. Cash paid at `t0` leaves the value: theta excludes cash flows. |

## 3. Design

### 3.1 The boundary rule and the API

`ObservationSchedule.time_shift(time_bump, bumped_date=None, *, keep_landing=False)`:

- a record before the bumped instant is dropped;
- a record ON it is kept at `observation_time = 0.0` when `keep_landing` and dropped otherwise.
  Numeric: `is_zero(t - time_bump)` (`Tolerance.ZERO`), so float residue never leaves a record at
  1e-17 (the QUAD time merge raises on `0 < t <= 1e-10`, `snowball_quad_engine.py:2425`).
  Dated: the same calendar date, `observation_date.date() == bumped_date.date()`, whatever the
  time of day (the bumped date keeps the valuation time of day, a fixing date is usually
  midnight; `calculate_year_fraction` raises on `end <= start`, so the kept dated record is
  written with an explicit `observation_time = 0.0`);
- the result is never `None`: a schedule with no record left is `ObservationSchedule(records=[])`.

`keep_landing` is passed through `BaseEquityOption.time_shift` and every product override
(one-touch, double one-touch, single and double sharkfin, accumulator, Snowball, Phoenix,
KO-reset through `BarrierConfig.time_shift`), and through `AutocallableCashLeg.time_shift`.
Product `time_shift` always ages the product (maturity or valuation date) and refreshes
`observation_dates` from the shifted schedule, `[]` included (today it refreshes them only
`if uses_times()`, so an emptied schedule would leave stale dates for the vol engines and
`_get_event_times`).

The theta paths call with `keep_landing=True`: numerical theta and `time_scenario`
(`riskmeasures/greeks/numerical.py`), `execution/greeks.py`, `portfolio/equity/position.py`,
`report/autocallable_risk_report.py`. Lifecycle aging, replay and the dashboard keep the default.
Intraday theta (`intraday/greeks.py:434`) does not call `time_shift` and is out of scope.

The `dropped_all` return value is removed: `time_shift` returns the carried outcome (3.2), and
the theta-is-zero branches in the five callers go. A bump at or past maturity keeps today's
handling (`current_maturity <= time_bump` returns 0 before any shift).

### 3.2 The outcome at `t0`

The decision at `t0` is the engines' own decision on the known spot, made once by the product:

- **Autocallables** (Snowball, Phoenix, KO-reset): `product.decide_observations_at_valuation(spot,
  env, knocked_in=state.knocked_in)` before the shift. Knocked out at `t0`: the bumped value is the
  KO payment's remaining receivable, through the existing terminal lifecycle state
  (`terminal_lifecycle_pv`). Knocked in at `t0`: the bumped price carries
  `AutocallableLifecycleState(knocked_in=True)`. Phoenix arrears released or lapsed at `t0` set the
  bumped product's carried arrears. A coupon decided at `t0` is cash at `t0` and leaves the value.
- **Barrier family** (barrier, double barrier, touch, sharkfins): a knock-out or touch at `t0`
  carries the terminal state (receivable if paid after `t1`); a knock-in at `t0` carries
  `BarrierLifecycleState(knocked_in=True)`; nothing breached carries nothing.
- **Accumulator:** the `t0` record is appended to `past_observations` as `(time, spot)`, its time
  written in the same convention as the existing past records (checked against
  `get_realized_accrual`, `accumulator_option.py:405-427`, in step 3), before the record is dropped.
- **Range accrual:** its time-based records are shifted like the others (today they are not),
  and a `t0` record becomes a past fixing with `observed_in_range` decided on the spot.

The theta paths pass the carried state as `lifecycle_state` to the bumped price. A caller that
already holds a lifecycle state at `t0` (portfolio positions, `execution/greeks`) passes it into
the `t0` decision, which updates it.

### 3.3 No observation left

A discretely monitored product with `records=[]` is a contract whose monitoring is over. The
schedule says so explicitly: `ObservationSchedule` gains `monitoring_ended: bool = False`, set only
by `time_shift` when it drops the last record. `ObservationSchedule.validate` and the products'
discrete-monitoring checks accept no records only when `monitoring_ended` is set; a new product,
or a schedule emptied by hand, still needs at least one record. The sharkfins stop regenerating past dates from the frequency
when the schedule is empty (`single_sharkfin_option.py:254-255`, `double_sharkfin_option.py:274-275`).

Each engine prices the remaining contract with no hit possible, before any path that needs a
record (the analytic shift needs two times; `assert_analytical_ready` and every product rebuild
re-validate):

| Product | Remaining contract |
|---|---|
| Knock-out (single, double) | participation × European (Black-Scholes, terminal payment delay) |
| Knock-in not knocked in | rebate × DF(T_pay); knocked in (carried state): participation × European |
| One-touch / double one-touch not touched | 0 |
| No-touch / double no-touch | rebate × DF(T_pay) |
| Single sharkfin | participation × (call spread K..B or put spread B..K) + no-hit cash × DF |
| Double sharkfin | participation × (spread to the barrier on the payoff side) + no-hit cash × DF |
| Accumulator | realized accrual (past observations) + terminal legs, discounted |
| Range accrual | accrued fraction × DF (already the engines' path when no future record remains) |

Engines and the code path to change (survey of 2026-10-07):

| Product | Analytical | MC | PDE | QUAD (discrete) | Vol MC / PDE |
|---|---|---|---|---|---|
| Barrier | before `DISCRETE` and `_price_beyond_discrete` | before the immediate-hit test (`barrier_option_mc_engine.py:197`) | `_setup_observation_indices` / `_valuation_instant_records` skip `resolve`; the adapter's replication hook prices the European; the knock-in composite skips the KO leg | `early_price` hook | `_monitoring` reads `observation_dates` (`barrier_vol_mc_engines.py:73`, `barrier_vol_pde_solvers.py:66`) |
| Double barrier | before `DISCRETE` | — | as barrier | `early_price` | — |
| One-touch | before `DISCRETE` | — | as barrier | `early_price` | — |
| Double one-touch | — | — | as barrier | `early_price` | — |
| Single sharkfin | before building legs | before `:141` | — | — | — |
| Double sharkfin | before `_shift_discrete_barriers` and leg rebuilds | before `:159` | — | — | — |
| Accumulator | guard `schedule.resolve` (`:224`) | guard `_resolve_observation_legs` (`:229`; `:160` already handles no times) | — | — | — |

The legacy autocallable QUAD engines raise on an empty KO or KI side
(`snowball_quad_engine.py:204-205`, `:226-227`; `phoenix_quad_engine.py:106-107`, `:125-126`;
`ko_reset_snowball_quad_engine.py:146`, `:158`, `:181`, `:566`), unlike their MC, PDE and QUAD V2
counterparts. An empty KI side already arises today when the knock-in dates end before the
knock-out dates; an empty side is folded into the time set as QUAD V2 does
(`quad/v2/adapters.py:160-193`).

The intraday twin builds `records=[]` once no marks remain (`intraday/twin.py:357-369`) and hits
the same engines; it is fixed by the same branches, with its own test.

## 4. What does not change

- Lifecycle aging, replay, backtest repricing, the dashboard: `keep_landing=False` and no outcome
  carry (they book `t0`/`t1` events themselves). Byte-identical.
- Prices: no engine's price of an unshifted product moves except where it raised on
  `records=[]`.
- Intraday theta, FX products (their `time_shift` drops nothing), DCN (its schedule builder
  already keeps the valuation date).

## 5. Verification

- **Boundary unit tests** for `ObservationSchedule.time_shift`: numeric and dated, landing kept
  or dropped by flag, float residue snapped to 0.0, time-of-day ignored, empty result is
  `records=[]`, a Friday-to-Monday bump drops Saturday and keeps Monday.
- **Theta against a hand recursion** on each family, at `t1` an observation date and not: the
  bumped value equals the product priced as of `t1` with the `t1` record at time 0. The daily
  accumulator's theta is restated with its number before and after.
- **Outcome carry:** at a `t0` knock-in (spot below the barrier, `t1` not a knock-in date) the
  bumped Snowball, Phoenix and KO-reset price as knocked in on PDE, MC and QUAD; the accumulator's
  `t0` shares are in `past_observations`; a `t0` knock-out's theta is the receivable's carry.
- **No observation left:** each engine × product cell of 3.3 against the closed form of its row,
  to round-off (analytic) or the engine's own tolerance.
- **Restated tests:** `test/test_greeks_theta_schedule.py` (`:68` drops the bumped-date record;
  `:136-175` asserts theta 0 on an emptied schedule; `:310`, `:337` the Monday record), and any
  test whose observation lands on a bump (`test_aged_accrual.py`, `test_phoenix_coupon_accrual.py`,
  `simulation/test_aged_contract.py`, `test_snowball_life_surface.py`,
  `test_pde_event_readout_consistency.py`, `test_pnlexplain_snapshot.py`,
  `test_equity_option_settlement_contract.py`, `test_autocallable_leg_time_shift.py`): each change
  listed with its before and after value.
- **Goldens:** replay theta columns (`test/replay_golden/data/scalar_bsm_greeks.csv`,
  `localvol_greeks.csv`; the fixture Snowball has KO at 2/365 and 5/365 and KI at 1/365 and 3/365)
  are regenerated in one commit with every moved row listed; `book_greeks.csv` carries no theta
  and stays byte-identical (lifecycle aging is unchanged). The PDE byte book's `bump_theta` fields
  are diffed base against branch with each mover declared. No certificate pins theta.
- **Full suite** under the RAM guard (at most 6 workers).

## 6. Order of work

1. `ObservationSchedule.time_shift` flag, snapping, empty result; product overrides pass the flag
   and always age (no caller passes it yet: byte-identical).
2. Empty-schedule acceptance and the engine branches of 3.3, each with its closed-form test.
3. The `t0` outcome carry (3.2) and the theta callers switched to `keep_landing=True`.
4. Restated tests and the golden regeneration, with the movers listed.
5. Changelog and docs (`docs/engine_param_guide.md` theta section if it states the old rule).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Theta moves on every product with an observation on the next date — every daily-monitored product, every day. | It is the ruled behaviour (R1); the changelog states it with the accumulator's numbers and the golden diff lists every moved row. |
| A `t0` decision in the theta path disagrees with the engine's own `t0` decision. | The carry calls the products' shared decision (`decide_observations_at_valuation`, the barrier family's valuation decision), the same code the engines read. |
| An engine branch for `records=[]` hides a schedule that was empty by mistake. | Engines take the branch only when `monitoring_ended` is set; validation refuses any other empty schedule. |
| Lifecycle aging gains the flag and someone passes `True` there. | The flag defaults to `False`; R2 is stated on the parameter, and a replay golden test guards the booking path. |
