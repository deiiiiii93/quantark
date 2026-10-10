# Run fixings: recording a fixing on the run day it passes — design

Status: **rev 2**, 2026-10-09, workstream W6, branch `w6-run-fixings-spec`, rebased onto `w5b-fills-unbumped` at
`20a02bc9` (w5b rebased onto main `33201993`). Rev 1 (`a38a615d` after the rebase, first `f3d0830a` on `b376cc18`) asked Q1–Q3; rev 2 folds in the
user's answers (§9) and re-measures every probe and every `file:line` citation on `20a02bc9`, where w5b has since
made a t = 0 fixing live (`6955cab7`, `is_history`) and made explain's pv_t1 the t1 snapshot's own value with a
`Factor.FIXING_REFILL` row (`20a02bc9`). No library code is changed by this spec.

Request (user, 2026-10-09): "Add a fixing module in backtest, scenario analysis, replay and PnL explanation." In a
multi-day run a fixing that passes DURING the run is never recorded; every later day re-decides it at that day's spot,
so a day-5 fixing is re-judged on day 20 at day 20's spot and the run books P&L that never happened. W5b left a
`TODO` at each place (`grep -rn "TODO" quantark | grep -i fix` on `20a02bc9`):

| Path | TODO site |
|---|---|
| equity backtest | `quantark/backtest/equity/engine.py:257` |
| FX backtest | `quantark/backtest/fx/engine.py:48` |
| equity dynamic scenario | `quantark/dynamicscenario/engine.py:117` |
| FX dynamic scenario | `quantark/dynamicscenario/fx/engine.py:69` |
| replay, pricing copy | `quantark/backtest/replay/product_replay.py:215` |
| replay, carry closures | `quantark/backtest/replay/product_replay.py:549` (`carry_price_callback`), `quantark/backtest/replay/engine.py:52` (`_book_price_callback`) |
| PnL explain | `quantark/pnlexplain/equity/explain.py:63` |

Evidence: `docs/superpowers/specs/2026-10-09-run-fixings-evidence/`. Every `.out` is a run on `20a02bc9` in this
worktree with `env PYTHONPATH=<wt>:<wt>/test /Users/fuxinyao/quant-ark/.venv/bin/python <script>` (cwd = scratchpad,
because the equity backtest writes a `logs/` directory into the cwd). Where a number changed between `b376cc18` and
`20a02bc9` the spec gives both.

| Script | Measures |
|---|---|
| `_fixtures.py` | shared contract and path (§2.1) |
| `probe_equity_runs.py` | E1 equity backtest, E2 equity dynamic scenario, E3 replay |
| `probe_fx_runs.py` | F1 FX backtest, F2 FX dynamic scenario |
| `probe_pnl_explain.py` | P1–P4 explain pairs around a fixing |
| `probe_live_fixing_greeks.py` | G1 a fixing dated today (t = 0) under the Greek freeze |
| `probe_legacy_asian.py` | L1 a legacy-schedule Asian has no past fixings |
| `probe_moved_explain_test.py` | M1 the w5b explain fixture: pv_t1 = the t1 snapshot's own value (done on `20a02bc9`) |

## 1. Binding decisions

### 1.1 Standing (before this spec)

- **History doesn't move** (W5b): a recorded or filled past fixing never takes a bumped or shocked spot (Greeks, VaR,
  stress, scenarios). A fixing missing at the first valuation of a run is filled once at that valuation's spot and
  kept (`quantark/util/fixings.py`: `is_history`, `freeze_missing_fixings`, `freeze_portfolio_fixings`; product
  protocol `missing_fixing_fills` / `with_fixings_recorded` / `with_missing_fixings_filled`).
- **Missing past fixings with no data**: filled with the current spot and a `MissingFixingWarning`, never raised.
- **Numeric == dated twin**: a numeric contract shows no difference from its dated twin.
- Products engine-agnostic; parameter objects immutable; QuantArk exceptions; `quantark/util/numerical` for every
  comparison; no invented fallback semantics (a `# TODO` and stop instead).
- **The real fixing of a run day is that day's market spot** — the QUOTED spot (`fx_env.spot` for FX, TARN spec Q3).

### 1.2 User decisions of 2026-10-09 taken into this spec

- **D1 — today's fixing is live.** A fixing at exactly the valuation instant (t = 0, today's unprinted fixing) moves
  with spot bumps and shocks; only strictly past fixings (t < 0) are frozen. *Done on w5b (`6955cab7`):
  `missing_fixing_fills` selects `is_history(t)` only.*
- **D2 — the clock fixes it.** In a simulation (dynamic scenario, backtest, replay), when time moves from t0 to t1 the
  fixing dated t0 is fixed at t0's spot — the run day's spot on the day it was live — and is history from t1 on.
  *This spec.*
- **D3 — explain's pv_t1 is the t1 snapshot's own value**, so consecutive daily explains add up. *Done on w5b
  (`20a02bc9`).*
- **D4 — a change in a filled missing fixing between t0 and t1 is its own labelled explain row.** *Done on w5b
  (`20a02bc9`, `Factor.FIXING_REFILL`) for fixings missing at t0; this spec extends it to fixings that pass inside the
  step (§4).*
- **D5–D7** — the answers to rev 1's Q1–Q3 (§9): gap fixings at the previous path day's spot; time-of-day path
  instants refused; explain fixings inside the step fixed at S0.

## 2. Measured current behaviour (on `20a02bc9`)

### 2.1 Fixture (`_fixtures.py`)

Start 2024-01-01; daily calendar path; spot 105 (inside the [90, 110] range) through day 4, then +1 a day (115 on day
14, 125 on day 24). Five fixings dated day 4, day 18, day 120, day 240, day 365. Range accrual: initial price 100,
accrual 10%, `initial_date` = start. Asian: fixed-strike arithmetic call, strike 100. Equity env flat vol 25%,
r 5%, q 2%, `CALENDAR_DAYS`. FX: EURUSD range [1.10, 1.30], spot 1.25 through day 4 then +0.01 a day, notional 1e6,
r_d 5%, r_f 3%, vol 10%. Numeric twins: the same records with `observation_time = days / 365` from the start.

Method: the run's own value on day k vs the engine's price on day k's environment of the same contract with every
fixing dated strictly before day k recorded BY HAND at its own day's spot (D1: the fixing dated day k is live and the
engine decides it at day k's spot). The replay builds its own environment (futures-implied carry), so its reference
is a replay over a window starting on day k.

### 2.2 Results (value of quantity 1; "run − hand" is the P&L the run invents)

| Path | Product | day 4 (fixing, spot 105 / 1.25) | day 10 | day 14 | day 24 | `MissingFixingWarning`s per run |
|---|---|---|---|---|---|---|
| E1 equity backtest | range accrual | 0 | **−1.905067** | −1.906111 | −1.908724 | 75 |
| E1 | Asian | 0 | **+0.953377** | +1.717028 | +4.846391 | 79 |
| E2 equity dynamic scenario | range accrual | 0 | −1.905067 | −1.906111 | −1.908724 | 128 |
| E2 | Asian | 0 | +0.953377 | +1.717028 | +4.846391 | 136 |
| E3 replay (analytical route) | Asian | 0 | +0.946551 | +1.710970 | +4.845753 | 47 |
| F1 FX backtest | FX range accrual | 0 | **−7,620.27** | −7,624.44 | −7,634.90 | 81 |
| F1 | FX foreign range accrual | 0 | −10,178.63 | −10,492.88 | −11,279.40 | 81 |
| F2 FX dynamic scenario | FX range accrual | 0 | −7,620.27 | −7,624.44 | −7,634.90 | 81 |
| F2 | FX foreign range accrual | 0 | −10,178.63 | −10,492.88 | −11,279.40 | 81 |

The values are bitwise those measured on `b376cc18`; only the warning counts rose (64–110 there), because the Greek
freeze no longer records the t = 0 fixing, so each bump call's pricing fills it again.

Reading:

- Days 0–4 agree to the printed digits; day 5 (spot 106, still in range) agrees for the range accrual because
  re-deciding gives the same outcome. From day 10 the day-4 fixing is re-decided OUT of range (range accrual) or at a
  higher price (Asian) every day: the run's value is wrong by about one fixing's coupon (range accrual) and by a
  growing amount (Asian: the day-4 price follows the spot).
- **Numeric twins do not age at all** in the equity backtest, both dynamic scenarios and the FX backtest (the
  `run numeric` column of every table: days 0–4 identical, `5.552080` equity range accrual, `28,832.3152` FX), so the
  fixing never passes. Equity: `PortfolioLifecycleManager._roll_untracked` reprices schedule-bearing numeric products
  with a constant maturity (`asset/equity/lifecycle/manager.py:181-198`); FX: only `env.valuation_date` moves (TARN
  spec M7). The replay ages a numeric product's `maturity` but not its records (`AutocallableLifecycleTracker.
  product_for_pricing`, `asset/equity/lifecycle/autocallable.py:219-265`): E3 numeric Asian 7.807415 vs dated 7.686469
  on day 4. **Numeric == dated is broken in every run path today, independently of fixings.**
- The quanto range accrual (`FxQuantoRangeAccrualOption`) subclasses `FxRangeAccrualOption` and shares its records and
  hooks; its engine needs a quanto environment, so it was not run. It behaves as the domestic one by construction.

### 2.3 PnL explain (`probe_pnl_explain.out`, waterfall + Taylor, dated range accrual)

| Case | pv_t0 | pv_alive_t1 | pv_t1 (= t1's own) | Result on `20a02bc9` |
|---|---|---|---|---|
| P1 t0 = day 4 (fixing at t = 0), t1 = day 10, t1 unrecorded | 5.707299 | 2.744374 | 2.744374 | runs; total −2.962925; spot row −3.126644; Taylor unexplained −1.977266; 16 warnings |
| P2 same, t1 records the fixing in range | — | — | — | **raises** `ValidationError: contract replacement is not a time step` |
| P2b same, t1 records it OUT of range | — | — | — | **raises** the same |
| P3 t0 = day 3, t1 = day 5 (fixing strictly inside), t1 unrecorded | 5.684708 | 5.592668 | 5.592668 | runs; 3 warnings |
| P3b same, t1 records it | — | — | — | **raises** the same |
| P4 fixing missing at t0 (strictly past), t0 spot 105, t1 = day 6 at 111 | 5.135560 | 4.833192 | 2.453163 | runs; `fixing_refill` row **−2.380029** |

- P4 is the w5b rule working (D3, D4): the shocked states keep the t0 fill, pv_t1 is t1's own value, the re-fill is
  its own row.
- P1, P3: a fixing that passes inside the step is not one of w5b's "missing at t0" fills, so every t1-dated state
  decides it at its own spot. In P1 the all-market state decides the day-4 fixing at 111 (out), so its effect lands in
  the spot row (−3.126644, of which Taylor leaves −1.977266 unexplained) instead of being fixed at t0's spot (D2).
- P2, P2b, P3b: a t1 contract that records a fixing the t0 contract left open is refused, because
  `observation_records` is fingerprinted with its outcome fields (`pnlexplain/equity/fingerprints.py:178-185`; records
  are not in `SCHEDULE_FIELDS`) and the roll check runs on the contracts as supplied (`explain.py:91-94`). The data a
  desk would book at t1 cannot be explained. On `b376cc18` P2 ran, because the old freeze recorded the t = 0 fixing at
  S0 on the t0 contract too; D1 removed that coincidence.

### 2.4 Today's fixing under the Greek freeze (`probe_live_fixing_greeks.out`)

Dated range accrual on day 4 at spot 109.9 (barrier 110), fixing #0 dated day 4. On `20a02bc9` `freeze_missing_fixings`
leaves it open (`None`) and `GreeksCalculator` delta is **−1.057904**, the live central difference with the
calculator's own 1% bump (on `b376cc18` the freeze recorded it and the delta was −0.191889). The probe still counts
6 `MissingFixingWarning`s: `resolve_observations` warns for a t = 0 fill (`range_accrual_option.py:529-534`, also
`asian_option.py:410-415`, `fx_range_accrual_option.py:270-275`), although under D1 that fixing is live, not missing
(§3.2).

### 2.5 Legacy Asian schedules (`probe_legacy_asian.out`)

An `AsianOption` with `exercise_date` + `num_observations` (no records) re-spreads its 12 observations over the
remaining life at every valuation (day 0: first at day 30.4; day 40: first at day 67.1; day 200: first at day 213.8;
always 0 past): it has no past fixing to record. Not covered (§5.2).

## 3. The module

### 3.1 Name and location

`quantark/util/run_fixings.py`, next to W5b's `quantark/util/fixings.py`. Asset-neutral and duck-typed like
`PortfolioLifecycleManager` (it never imports `quantark.portfolio`): it acts on any object with `positions` (each with
`product`, `quantity`, `underlying`) and `pricing_environments`. Public names:

```python
class RunFixingRecorder:
    def register(self, portfolio) -> RunFixingDay: ...     # the run's first valuation
    def advance(self, portfolio) -> RunFixingDay: ...      # every later path instant, after the market is applied

def step_fixings(product, env, *, prev_date, prev_spot, run_fills,
                 live_spots=None) -> Tuple[Any, Dict[int, Any]]: ...
    # stateless core used by the recorder and by ProductReplay (§3.6); live_spots: amendment below

def settle_history(product, env) -> Any: ...   # optional product hook, amendment below

@dataclass(frozen=True)
class RecordedFixing:
    position_id: str
    index: int            # record index (records are never dropped by these products)
    label: str            # W5b ``fixing_label``
    outcome: Any          # bool (range accrual) or price / rate (Asian, TARN, TARF)
    spot: float           # the spot it was fixed at
    gap: bool             # True when no path instant fell on its date (rule R4)

@dataclass(frozen=True)
class RunFixingDay:
    date: datetime
    fixings: Tuple[RecordedFixing, ...]
    product_events: Tuple[Tuple[str, float, Any], ...] = ()   # (position_id, quantity, event): TARN/TARF cash (§6)
```

**Amendment 2026-10-10 (as built in W4b Task 9 fix round 1, controller ruling; W6 participants byte-identical).**
`step_fixings` takes `live_spots` (`{index: spot of the first instant the record was live}`, kept by the recorder
with `update_live`). It records the LIVE entries first, in index order, each at its first-live spot (R3 / Ruling 4;
consecutive indices sharing one spot in one `missing_fixing_fills` probe), and only then probes the recorded copy at
`prev_spot` for the remaining (gap, R4) records. Before, it probed the whole listing at `prev_spot` and re-decided only
the live entries, which is the same for W6's range accruals and Asians (their open records do not depend on earlier
outcomes) but wrong for a TARN / TARF on a business-day clock whose weekend rows quote a spot other than Friday's (a
redemption at Friday's spot kills later fixings). `settle_history(product, env)` calls the optional product hook
`with_history_settled(env)` (W4b TARN / TARF: bank the recorded strictly past fixings and pay the cash whose pay time
is strictly past, on a copy; a product without the hook is returned unchanged). The recorder calls it once at a
position's registration, after the W5b fill (its events are not reported), and at every clock-moving `advance` right
after the step's recording (its FIXING / PAYMENT events go to `RunFixingDay.product_events`); a same-date `advance`
(Ruling 9) does not call it. `ProductReplay` does not settle product-held history and refuses a product with the hook
(`ValidationError`, `TODO(tarn-replay)`).

### 3.2 Product protocol (extends W5b)

A product takes part when it has all three methods below; every other position is left exactly as today.

| Method | Status | Meaning |
|---|---|---|
| `missing_fixing_fills(env) -> Dict[int, outcome]` | W5b, strict since `6955cab7` | every record with `is_history(t)` (t < 0, not zero) without an outcome, decided at `env.spot` (`range_accrual_option.py:549`, `asian_option.py:437`, `fx_range_accrual_option.py:383`). Unchanged by W6. |
| `with_fixings_recorded(fills) -> copy` | W5b, unchanged | records the listed outcomes on a copy; a record already carrying an outcome keeps it. |
| `with_time_elapsed(dt) -> copy` | **new** | the clock step of a numeric contract, recording nothing: every numeric record time, `maturity` (and the range accrual's `accrued_offset`, as `time_shift` banks it) move by `dt`; a shifted time with `is_zero` becomes exactly `0.0` (so a twin's fixing lands at t = 0 on its day, not an ULP either side); a dated contract returns `self`. A `num_observations` / `observation_times` schedule is written out as records first (as `RangeAccrualOption.time_shift` does). |

Also changed by W6, in the same products' `resolve_observations`: a record at t = 0 without an outcome is still
decided at the current spot (that is what makes it live), but it no longer counts as "missing": `MissingFixingWarning`
names only `is_history` fills (§2.4).

Why a new clock method instead of `time_shift`: the equity `RangeAccrualOption.time_shift` records every record that
becomes past OR lands on the bumped instant at the held spot (`range_accrual_option.py:404-439`) — correct for theta,
wrong for a run, where D1 keeps the landing fixing live at the new day's spot. `AsianOption` has no record-moving
`time_shift` at all (the base class moves `maturity` only, `base_equity_option.py:649-669`). The FX `time_shift`
(`fx_range_accrual_option.py:360`) moves times without recording but mutates and does not snap to 0.0. One
copy-returning method gives all of them the same run semantics without touching theta.

Implemented by: `RangeAccrualOption`, `AsianOption` (record schedules only; a legacy schedule does not take part, §5.2),
`FxRangeAccrualOption` and its subclasses `FxQuantoRangeAccrualOption`, `FxForeignRangeAccrualOption`; TARN/TARF per
§6.

### 3.3 Run-day semantics

- **R1 Which spot.** The run's market spot at the path instant: equity `pricing_env.spot` after the day's market is
  applied; FX the quoted `fx_env.spot` (never `effective_spot()`); quanto / foreign range accruals: the pair's quoted
  spot (the rate the record observes).
- **R2 Live vs history (D1).** At a valuation, a record at t = 0 without an outcome is live: the engine decides it at
  that valuation's spot, bumped or shocked if the valuation is. A record at t < 0 is history: recorded (by the run or
  by the booking) or, at the first valuation only, filled (R5).
- **R3 The clock fixes it (D2).** When the run's clock moves from path instant P (date `d_P`, spot `S_P`) to the next
  path instant P′, every record that is strictly past at P′ and has no outcome is recorded at **`S_P`**. Those are
  exactly the records dated `d_P` (live at P) and the records dated strictly between `d_P` and `d_P′` (R4). Records
  dated `d_P′` stay live at P′. **Amendment (user approved 2026-10-09):** a record that was live (t = 0) at more than
  one instant — a Friday fixing on a `BUSINESS_DAYS` clock with weekend path rows is t = 0 on Friday, Saturday and
  Sunday — is recorded at the spot of the FIRST instant it was live (Friday here; in general the first instant can fall after
  the record's date, e.g. Thursday then Saturday rows), not at `S_P` (Sunday). On calendar-day clocks a record is live at exactly one instant, so this is the rule above.
- **R4 Holidays and gaps (D5).** Path instants are whatever the path holds (business days, `W`/`M` equity backtest
  frequencies, sparse `DayPath`s, and every numeric `k/12` fixing on a daily path). A record dated strictly between two
  path instants was never live on the path; R3 records it at the spot of the PREVIOUS path instant, flags `gap=True`,
  and emits no warning (D2 extended to a date the path skips; the theta rule — a bump fixes the fixings it steps over
  at the held spot).
- **R5 The first valuation.** W5b's rule: records strictly before the run's first valuation and without an outcome are
  filled once at that valuation's spot with one `MissingFixingWarning` per position (`freeze_portfolio_fixings`).
  Records dated the first day are live on it (R2) and are recorded at its spot when the clock moves (R3), without a
  warning.
- **R6 Numeric vs dated.** A dated contract ages by its environment's date. A numeric contract's times are years from
  the run's first valuation date (its clock origin, recorded at `register`; the TARN spec's ruling R3 and the equity
  `FloatMaturityRoller`'s first-sight anchor). Each step calls `with_time_elapsed(dt)` with
  `dt = calculate_year_fraction(d_P, d_P′, env.day_count_convention, env.bus_days_in_year, calendar=env.calendar)`,
  the environment's convention, so a numeric twin moves exactly as its dated twin resolves (TARN spec Q7 / §7.2; daily
  steps telescope to the direct year fraction to ≤ 5e-16, TARN M10). Then R3 applies to both forms identically: same
  record indices, same `S_P`, same outcomes bitwise; prices agree to relative 1e-12 (time ULPs).
- **R7 Undated runs.** A `DayPath` without `start_date` never moves the environments' date (TARN M8): nothing is
  aged and nothing passes, so the recorder is inactive (no refusal for these products; TARN keeps its own Q8 refusal,
  §6).
- **R8 Valuation instants are dates (D6).** A dated record is a calendar date at midnight. If a run's path instant
  carries a time of day (the equity backtest's `H` frequency, or an adapter stamping 15:00 closes), a record dated that
  day is already strictly past at the day's own instant and R3 would fix it at the previous day's spot. That needs a
  fixing time on the record, which no product has. **The recorder raises `ValidationError` when a participating
  product is in a run whose path instant is not midnight** (`# TODO(run-fixings): intraday fixing instants — a
  fixing-time term on the records`, a follow-up, §10).
- **R9 Clock direction.** A path instant earlier than the previous one raises `ValidationError` (as TARN §7.2).
- **R10 What is recorded on the product.** The run's working copy of each position gets
  `product.with_fixings_recorded(fills)`; the booked product is never mutated (W5b rule). Recorded outcomes are history
  in every later valuation, bump and shock of the run.

### 3.4 Interaction with the W5b freeze and with Greeks during a run

- `register` IS W5b's first-valuation freeze plus remembering `(d_P, S_P)` per underlying and each numeric position's
  clock origin; the call sites that call `freeze_portfolio_fixings` today call `register` instead.
- After `advance`, no participating product has a strictly past record without an outcome, so every Greek computed in
  the run (`GreeksCalculator`, `BaseFxEngine.calculate_greeks`, `get_portfolio_greeks`) and every replay carry closure
  finds nothing to freeze and emits no `MissingFixingWarning` (with the §3.2 warning change). Today's fixing (t = 0)
  is live under the bumps (D1; G1).
- Theta inside the run predicts the run: a 1-day theta bump moves the clock and fixes the stepped-over fixing at the
  held spot (`RangeAccrualOption.time_shift`, `BaseFxEngine._fdm_theta` + `record_passed_fixings`) — the same spot R3
  uses the next day.

### 3.5 Run-engine integration (line numbers on `20a02bc9`)

| Path | Today | Change |
|---|---|---|
| Equity backtest | `engine.py:253` `_update_pricing_environment`; `:254-260` first-step `freeze_portfolio_fixings`; `:264` explain `begin_day`; `:272` `_process_lifecycle`; `:300` Greeks | `:254-260` become `self._fixings.register(self.portfolio)` on the first step and `self._fixings.advance(self.portfolio)` on every later one, both before `:264` so the explain recorder's alive product already carries the step's recordings. `PortfolioLifecycleManager._roll_untracked` (`manager.py:181-198`) skips participating products (no constant-maturity warning; the recorder ages them). |
| FX backtest | `engine.py:45` first `_apply_row`; `:50` freeze; `:64` per-row `_apply_row` | `:50` → `register(portfolio)`; after `:64` → `advance(portfolio)` (a no-op for the first row, same date). |
| Equity dynamic scenario | `engine.py:119` freeze of the INPUT portfolio; `:159` baseline; `:182` day changes; `:187` env date; `:192` lifecycle | `:119` removed; `register(working_portfolio)` before `:159`; after `:187`, when `day_date` is not None, `advance(working_portfolio)` before `:192`. |
| FX dynamic scenario | `engine.py:71` freeze of the input; `:86` baseline; `:92` day changes; `:95` env date | `:71` removed; `register(working)` before `:86`; after `:95`, when `day_date` is not None, `advance(working)`. |
| Replay | `product_replay.py:210-225` `product_for_date` (W5b `_run_fills`); `engine.py:409` (first-day initial price), `:439` (daily price); carry closures `product_replay.py:535-552` and `engine.py:43-55` freeze the day's contract at the day's env | `product_for_date` calls `step_fixings` (stateless, §3.6) whenever `date` is later than the last date it saw, growing `_run_fills`; the first call stays the W5b freeze. The carry closures keep their `freeze_missing_fixings` call (after R3 it finds nothing strictly past; the t = 0 fixing stays live under the carry bumps) and lose their TODOs. `AutocallableLifecycleTracker.product_for_pricing` (`autocallable.py:219-265`) ages a participating non-autocallable product with `with_time_elapsed(elapsed)` instead of `maturity -= elapsed`, so its records move with its maturity (E3 numeric). |

The two dynamic-scenario engines freeze the input portfolio today and then clone it; registering on the working copy
keeps the clock origin at the date the run values first, and the input portfolio is never touched.

### 3.6 The replay is stateless per date

`ProductReplay` rebuilds the pricing copy each day from the tracker's original contract. It keeps
`_run_fills: Dict[int, outcome]` (exists, W5b) plus `_last: (date, spot)`. On a date later than `_last.date`:
`product = tracker.product_for_pricing(date)`; `step_fixings(product, env, prev_date=_last.date,
prev_spot=_last.spot, run_fills=_run_fills)` computes `product.missing_fixing_fills(env with spot = prev_spot)` (the
records that became history in the step), adds them to `_run_fills`, and returns
`product.with_fixings_recorded(_run_fills)`. (Amendment 2026-10-10, §3.1: as built, the replay also keeps `_live`
(`update_live`) and passes it as `live_spots`, so `step_fixings` records the live records first at their first-live
spot and then probes the gaps at `prev_spot`; it never calls `settle_history` and refuses a product carrying
`with_history_settled`.) The second call on the same date (the replay calls `product_for_date`
twice on its first date, `engine.py:409` and `:439`) records nothing new. The replay prices post-close; for the
autocallables its tracker applies a record due today at today's close — that rule is the tracker's, not changed here;
range accrual and Asian records follow R2/R3.

## 4. PnL explain

### 4.1 What the pair knows

Two snapshots: the t0 contract and environment, the t1 contract and environment. No market path between them. A
fixing that becomes history inside the step has one honest source of truth — the t1 contract, if it records it. If
it does not:

- a fixing dated t0 was live at t0; D2 fixes it at **S0** (t0's spot);
- a fixing strictly inside (t0, t1) has no observation in the pair; its real value is **unknowable** from the
  snapshots. D7: the pair fixes it at S0 (the run rule R3/R4); its difference from the t1 snapshot's own decision (a
  W5b fill at S1, or t1's record) shows in the re-fill row;
- a fixing dated t1 is live at t1 (D1): it moves with the t1-dated states' spot.

### 4.2 What w5b already does (`20a02bc9`)

`explain` (`explain.py:83`) runs the pair and roll checks on the contracts as supplied (`:91-94`), then
`_freeze_pair_fixings` (`:50-80`) takes t0's `missing_fixing_fills` (strict), freezes the t0 contract, carries the fills
onto `transition.product_alive_t1` and onto a `snapshot_t1_carried`; pv_t1 is the t1 snapshot's own value
(`ScenarioCache.value_t1`); `event_row` (`lifecycle.py:215`) measures against `value_t1_carried`; `fixing_refill_row`
(`lifecycle.py:227`, `Factor.FIXING_REFILL`, "missing fixing re-filled") = `value_t1 − value_t1_carried`, emitted only
when t0 had a missing fixing.

### 4.3 Rules (W6 extends the carried decisions; no new factor)

- **E1 pv_t0** = the t0 snapshot's own value (unchanged).
- **E2 The pair's decisions.** For a participating product, the decisions carried onto `product_alive_t1` and
  `snapshot_t1_carried` become: t0's W5b fills (as today) PLUS every record strictly past at t1 that is open in the t0
  contract, decided at S0 (`missing_fixing_fills(env_t1 with spot S0)`, after `with_time_elapsed` for a numeric
  contract) — the fixings that pass inside the step, decided once for the pair, never by a shocked state's spot.
  Records dated t1 stay live. t0-dated states keep the t0 contract (its t = 0 record live at the state's spot).
- **E3 pv_t1** = the t1 snapshot's own value (unchanged, D3).
- **E4 The re-fill row** keeps its formula, `value_t1 − value_t1_carried`, and its factor `FIXING_REFILL`; it is
  emitted whenever E2 carried any decision, and its label/docstring become "a fixing the pair decided (at t0's spot)
  that the t1 snapshot decides differently". It is zero when the t1 contract (or its own fill) agrees with the pair's
  decision (D4).
- **E5 Contract check.** `check_contract_roll` (`fingerprints.py:312-368`) accepts an outcome field
  (`observed_in_range`, `observed_price`) that is open in the t0 contract and set in the t1 contract on a record
  strictly past at t1; that is a fixing, not a contract replacement, and E4 reports any disagreement with the pair's
  decision. A record the t0 contract RECORDED and the t1 contract records differently is still refused (a booked
  fixing changed: not a time step).
- **E6 Runs.** In the equity backtest and replay explain recorders the t1 contract carries exactly the run's R3
  recordings (made at S0 = the previous path day's spot), so the re-fill row is 0.0 there (test X3i).

### 4.4 What it gives on the probes

| Case | on `20a02bc9` | after W6 |
|---|---|---|
| P1 (fixing dated t0, t1 unrecorded) | pv_alive_t1 2.744374, pv_t1 2.744374, no re-fill row (spot row carries it) | pv_alive_t1 **4.649441** (fixed at S0, in range), pv_t1 2.744374, `fixing_refill` 2.744374 − 4.649441 = **−1.905067**; total unchanged −2.962925 |
| P2 (t1 records it in range) | raises | pv_t1 4.649441, re-fill 0 |
| P2b (t1 records it out of range) | raises | pv_t1 2.744374, re-fill **−1.905067** |
| P3 (strictly inside; S0 105 and S1 106 both in range) | pv_t1 5.592668 | pv_t1 5.592668, re-fill 0 |
| P3b (t1 records it in range) | raises | pv_t1 5.592668, re-fill 0 |
| P4 (missing at t0, t1 refills at 111) | re-fill −2.380029 | unchanged |

pv_alive_t1 and pv_t1 after W6 are measured values (the hand-recorded contract at t1: `probe_pnl_explain.out` P2 on
`b376cc18` and P1 on `20a02bc9`); the market rows of P1 also move (the all-market state no longer decides the fixing
at 111) and are not measured here.

## 5. Products

### 5.1 Covered now

`RangeAccrualOption` (numeric records, `observation_times`, `num_observations`, dated records); `AsianOption` with
`observation_records` (numeric or dated); `FxRangeAccrualOption`, `FxQuantoRangeAccrualOption`,
`FxForeignRangeAccrualOption`; FX TARN / TARF through §6.

### 5.2 Past-observation state NOT covered (list for later decisions)

| Product | Its past state | Status in runs |
|---|---|---|
| Snowball, Phoenix, KO-reset snowball | KO/KI/coupon records | equity lifecycle trackers (`asset/equity/lifecycle/`) already record them per run day; unchanged |
| Equity `BarrierOption`, `DoubleBarrierOption`, sharkfins, one-touch, double one-touch | touch history | `BarrierLifecycleTracker` only when `handle_lifecycle_events=True`; otherwise a touch is forgotten |
| `AsianOption` legacy schedules (`observation_times` / `observation_dates` / `num_observations`, no records) | none representable (L1: re-spread over the remaining life) | `# TODO`: needs records; the module does not take part |
| `AccumulatorOption` | `past_observations` (realized `(time, spot)`) | not recorded during runs |
| `DcnOption` | `knocked_in_at_valuation`, coupon observations | not recorded during runs |
| FX `FxBarrierOption`, `FxSharkfinOption`, `FxOneTouchOption` | touch history | frozen numeric in FX runs (TARN spec §9); no tracker |
| Equity TRS | realized cashflows | its own realized-cashflow accounting (TRS merge); not a fixing |

Numeric FX vanillas, digitals and barriers are still frozen in FX runs (TARN spec §9); this spec ages only products
with fixings.

## 6. FX TARN / TARF: subsume, do not sit beside

**Decision: the general module subsumes the TARN hook. TARN/TARF become products that implement the run-fixings
protocol; `FxTargetRedemptionLifecycle` is not built.**

Reasons:

1. **Same loop.** The planned hook (TARN plan Task 9, `register` / `advance`) is this module's loop: fill fixings
   before the run at the first spot with one warning, age numeric contracts by the environment's year fraction since
   the last date, record the fixings that have passed at a run spot. Two loops in one FX run would hold two clocks and
   two copies of the ordering rules, and a book with a range accrual and a TARN could age them differently.
2. **D1/D2 change the hook's rules anyway.** The plan banks "every fixing at or before `date` at that day's spot"
   (TARN spec §7.2 step 3) and banks a fixing on the origin date silently at registration (§7.2 register step 3). D1
   makes both live on their day; D2 fixes them at the day they were live. The hook would have to be rewritten to the
   rules of §3.3 regardless.
3. **One gap rule.** On a daily path every numeric `k/12` TARN fixing falls between path days; the TARN plan banks it
   at the next day's spot, D5 at the previous one. One recorder, one answer.
4. **The FX follow-up has one home.** The TARN spec's §9 (other numeric FX products frozen) and Q8 follow-up
   (undated paths) extend this recorder, not a TARN-only class.

What stays TARN-specific: the realized state, `resolve()`, pending cash and PV, terminal bookings, the cash columns,
the FIXING snapshots, `MaturedBeforeRunWarning` removal, and the Q8 refusal of undated paths.

**Concrete changes to the TARN plan (`docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle.md`):**

- **Prerequisite:** W6's library tasks (`with_time_elapsed`, the t = 0 warning change, `RunFixingRecorder`, call
  sites) land before Tasks 9–11 resume. Tasks 1–8 stand, with one fix to the built Task 6/7 `resolve()`: an unbanked
  fixing at t = 0 is still banked on the copy at the quoted (possibly bumped) spot — that is what makes it live — but
  only `is_history` fixings are named in the `MissingFixingWarning` (D1).
- **Task 9 →** "TARN/TARF implement the run-fixings protocol". No `quantark/asset/fx/lifecycle/`. Add to
  `TargetRedemptionProduct`: `missing_fixing_fills(fx_env)` = `{i: fx_env.spot}` for unbanked fixings with
  `is_history(t)`; `with_fixings_recorded(fills)` = a copy that banks those fixings IN ORDER at the given rates through
  the existing `record_passed_fixings` arithmetic (clip, redemption, dead fixings, PAYMENT events);
  `with_time_elapsed(dt)` = copy + `time_shift(dt)` + `is_zero` snap; keep `matured_before(fx_env)`. The recorder
  calls `matured_before` at `register` (warn `MaturedBeforeRunWarning`, remove the position — a protocol hook any
  product may implement), and reports the events each product appended during the step in
  `RunFixingDay.product_events`, plus `Σ quantity × pending_pv(env)` for products exposing `pending_pv`. Test changes
  in the plan's Task 9 list: `test_fixing_on_the_origin_date_banks_without_a_warning` → the origin-day fixing is NOT
  banked at `register` (`state.as_of` stays None), it banks on the first later `advance` at the origin day's spot,
  still without a warning; `test_d2_twins_age_identically` → on `FIX_DATES[1]` the dated fixing is live (not banked),
  banked the next day, numeric twin identical (snap); `test_calendar_gap_banks_in_order_without_a_warning`,
  `test_short_position_books_signed_cash`, `test_two_pairs_age_independently` → the banking spot is the previous path
  day's (flat paths: same numbers; `FIXING.fixing` = previous day's spot); `test_advance_needs_the_environment_on_that_date`
  and `test_clock_cannot_go_back` move to the recorder (R9).
- **Task 10 (FX backtest) →** keep the `paid_cash` / `pending_pv` columns and the cumulative-cash value; read them
  from W6's `RunFixingDay` (`advance` already installed after `engine.py:64`). B1–B6 stand on flat paths; change
  `test_business_day_path_banks_weekend_fixing_next_day`: a Saturday fixing is banked on Monday's advance at FRIDAY's
  spot with `gap=True` (R4/D5), not at Monday's.
- **Task 11 (FX dynamic scenario) →** keep Q8 (raise on an undated path when the book holds a TARN/TARF, before
  `register`), the FIXING snapshots (`spot` = the recorded rate = previous path day's spot) and the cash fields, fed
  from `RunFixingDay`. `test_dynamic_tarn_redeems_and_books_cash`: numeric `k/12` fixings already fall between days,
  so the event days `[31, 61]` and the amounts stay on the held path.

## 7. Tests

New file `test/test_run_fixings.py`; explain tests added to `test/test_missing_fixings_followup.py` (w5b's explain
re-fill tests live there). "Relative 1e-12" = `is_close(a, b, rel_tol=1e-12)`.

**Per path (X1)** — for each of the equity backtest, FX backtest, equity dynamic scenario (dated path), FX dynamic
scenario (dated path) and replay, with the §2.1 fixture (dated range accrual, dated Asian; FX: domestic and foreign
range accrual):
- X1a the run's value on every day k ≥ 5 equals the engine's price at day k's environment of the contract with the
  fixings strictly before k recorded by hand at their own day's spot (bitwise for the analytical engines; the replay
  against a replay window starting at k). Today: §2.2.
- X1b the day-4 fixing does not move later: re-running with a path that differs from day 6 on gives the same recorded
  outcome and the same `RecordedFixing.spot` (105 / 1.25).
- X1c no `MissingFixingWarning` is emitted after the first valuation (today: 47–136 per run).

**Numeric vs dated (X2)** — the numeric twin and the dated contract in the same run, every path: identical
`RecordedFixing` sequences (index, outcome, spot, gap) bitwise; values relative 1e-12 on every day; on a `BUSINESS_DAYS`
environment too (R6).

**Rules (X3)**
- X3a R3/D2: a fixing dated day 4 is live on day 4 (delta at 109.9 equals the live central difference with the
  calculator's bump: −1.057904 in G1's setup) and is history from day 5 at day 4's spot.
- X3b R4/D5: a weekly (`W`) equity backtest and a `DayPath` with a missing day: a fixing between path instants is
  recorded at the previous instant's spot with `gap=True`, no warning.
- X3c R5: a fixing strictly before the run → one `MissingFixingWarning` per position, filled at the first spot; a fixing
  dated the first day → no warning, recorded at the first day's spot on the step to day 1.
- X3d R7: an undated `DayPath` records nothing and values exactly as today.
- X3e R8/D6: a participating product with a path instant at 15:00 (and an `H`-frequency equity backtest) raises
  `ValidationError`; a book without one runs unchanged.
- X3f R9: a backwards path raises.
- X3g the booked products and the input portfolio are unchanged after every run.
- X3h Greeks inside a run equal `GreeksCalculator` on the hand-recorded contract at that day's environment.
- X3i explain recorders (equity backtest, replay): the re-fill row is exactly 0.0 every day, and
  `pv_t1(day N) == pv_t0(day N+1)`.

**Warnings (X4):** `resolve_observations` names only `is_history` fills, for every covered product (a t = 0 fill is
silent); `missing_fixing_fills` / `freeze_missing_fixings` keep excluding t = 0 (w5b, regression).

**PnL explain (X5):** P1–P4 per §4.4: pv_alive_t1, pv_t1 and the re-fill row equal the listed values; waterfall and
Taylor reconcile; P2, P2b and P3b no longer raise; a t1 contract that changes a t0-RECORDED outcome still raises; a pair
with nothing missing and nothing passing emits no re-fill row (as today).

**Hand equality (X6):** for each path, the run's final product equals (record by record) the booked contract with
`with_fixings_recorded` of the hand fixings.

## 8. Moved values (for sign-off)

Estimated by the probes on `20a02bc9` and by grep of `test/` for the covered products inside runs (only
`test/test_missing_fixings_scenarios.py` holds them in a run). D1, D3 and the P4 re-fill already moved on w5b and are
not repeated here.

| What | Before → after | Existing test |
|---|---|---|
| Dated range accrual / Asian / FX range accruals in a run, days after a fixing passes | §2.2 "run − hand" goes to 0 (e.g. equity range accrual day 10 2.744374 → 4.649441; FX 11,385.19 → 19,005.46) | none asserts (the w5b run tests' fixings are at day 181+; their runs are 4–8 days) |
| Numeric range accrual / record Asian in the equity backtest, both dated dynamic scenarios, FX backtest | frozen → aged (equals the dated twin) | `test_missing_fixings_scenarios.py::test_equity_dynamic_scenario_fills_once_at_the_first_valuation` and `::test_fx_backtest_fills_once_at_the_first_valuation` hold numeric contracts: both arms age identically, the assertions (got == want) hold, the values move (not asserted) |
| Numeric record Asian in the replay | maturity aged, records frozen → both aged (E3 day 4 numeric 7.807415 → the dated 7.686469, relative 1e-12) | none |
| The equity backtest's "no roll rule" `UserWarning` | fires for numeric range accrual / record Asian → no longer (they are aged) | `test_float_maturity_roll.py` uses a legacy-schedule Asian (not a participant): unchanged |
| `MissingFixingWarning` for a t = 0 fill | warned → silent | `test_missing_fixings.py::TestEquityRangeAccrualFill::test_near_expiry_fills` (both engine ids) asserts `len(caught) == 1` → 0 (price unchanged) |
| PnL explain with a fixing that passes inside the step (P1) | pv_alive_t1 2.744374 → 4.649441; no re-fill row → `fixing_refill` −1.905067; market rows move; total unchanged | none (w5b's `test_pnl_explain_shocked_states_keep_the_t0_fill` has no fixing inside its 3-day step: unchanged) |
| P2 / P2b / P3b explain | raises → explains | none |
| An `H`-frequency (or time-stamped) equity backtest holding a covered product | runs → `ValidationError` (R8/D6) | none found |
| TARN plan Tasks 9–11 expectations | per §6 | plan tests, not yet built |

## 9. Decisions on rev 1's questions (user, 2026-10-09)

- **D5 (Q1 a) Gap fixings.** A fixing dated between two path instants (weekly/monthly backtests, sparse `DayPath`s,
  numeric `k/12` fixings on a daily path) is fixed at the PREVIOUS path instant's spot, with no warning and
  `gap=True` (R4).
- **D6 (Q2 a) Time-of-day path instants.** A run whose path instant carries a time of day and that holds a covered
  product is REFUSED with `ValidationError` and a `# TODO` (R8). A fixing-time term on the records is a future
  follow-up (§10).
- **D7 (Q3 a) Explain fixings inside the step.** A fixing strictly inside (t0, t1) is fixed at S0 for the pair's
  revaluations; its difference from the t1 snapshot's own S1-fill (or t1's record) shows in the re-fill row (§4.3 E2,
  E4).

## 10. Out of scope and follow-ups

- **Follow-up (D6): a fixing-time term on observation records** (e.g. a close fixing at 15:00), so a run with
  time-stamped or intraday path instants can say when a dated fixing prints; until then such runs refuse covered
  products.
- Products in §5.2; FX PnL explain (none exists: `quantark/pnlexplain/` is equity only); cash for range accruals or
  Asians at expiry inside runs (unchanged); the autocallable trackers' post-close rule.
