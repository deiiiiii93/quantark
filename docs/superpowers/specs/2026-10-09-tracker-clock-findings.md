# Tracker clock findings: a numeric contract whose `initial_date` precedes the run

Date: 2026-10-09. Branch `tracker-clock-findings`, from `ko-reset-lifecycle-v2` at b419bfc7.
Status: **implemented (A)** on branch `w1-tracker-clock` (from main dcea77b6), with decision 2 (records dated before
the run). See section 6. The text below is the original findings and decision request; the option patches it
describes exist only as a runtime pytest plugin in the evidence directory.

Evidence: `docs/superpowers/specs/2026-10-09-tracker-clock-findings-evidence/`
(`measure_clock_mismatch.py` + `measure_clock_mismatch*.json`, `dated_catch_up.py`, `tracker_clock_options.py`,
`run_arm.sh`, `runs/`).

## 1. The mechanism

`AutocallableLifecycleTracker` (`quantark/asset/equity/lifecycle/autocallable.py`) runs **two clocks** for a
contract with numeric timing (`_uses_date_timing(product)` is False: numeric observation times, numeric maturity):

| Clock | Origin | Where | What runs on it |
|---|---|---|---|
| Event clock (record dates) | `initial_date or start_date` | `_record_dicts` :668 (date = base + `round(t*365)` days, :682); `_terminal_payment_date` :609; `_schedule_resolution_env` :635 | the day each KO / KI / coupon record is due (`_record_is_due` :765 compares the close with the record's date), its settlement date, the terminal payment date |
| Pricing clock | `start_date` | `product_for_pricing` :134 / :139 (`elapsed = (date - start_date)/365`); `_valuation_point` :716-729 | the pricing copy's ageing (maturity, `time_shift` of the schedules, the accrual); the numeric terminal's due day (`_timing_is_due` :772 uses the valuation point); the ledger's `determination_time` (:816, :847) |

KO-reset: `KOResetLifecycleTracker._schedule_two_base` (`ko_reset.py`:146) and `knock_in_time` (:154) put a REBASED
schedule 2 on the **pricing** clock (b419bfc7). Schedule 1 and the KI schedule stay on the event clock.

The simulated-path backtest copies the pricing clock: `backtest/simulation/lifecycle.py`:109 `elapsed_time` and
`backtest/simulation/pricing/surface.py`:260 `elapsed` both count from the run's first date, while
`resolve_calendar_schedule` dates the records on the event clock.

When `initial_date == start_date` the two clocks are the same clock and nothing below happens. When `initial_date`
is N days earlier, the tracker is wrong in four ways (measured in section 2):

1. **The copy lags the events by N days.** A record is booked N days before the copy stops holding it, so for N
   days after each observation the copy still prices an observation the tracker has already tested.
2. **Records before the run all fall due on the first close.** A record whose contract day is <= N is dated before
   `start_date`. It is tested against the run's first close, while the day-0 copy still prices it as a future
   observation.
3. **The tracker disagrees with itself at the end.** The last record is booked on day 365 - N, but the numeric
   terminal falls due on the pricing clock (day 365). The contract stays alive for N days after its final
   observation. During those days it keeps monitoring a continuous KI on every close, and it then pays its terminal
   off a close N days late.
4. **The ledger mixes the two clocks.** A knock-out booked on the event clock records its `determination_time` on
   the pricing clock. With N = 30 or 100, record 0 (contract time 30/365) knocks out with `determination_time = 0.0`.

## 2. Measurements

Contracts: a numeric snowball (KO 103, continuous KI 75, 15% annualized) and a numeric memory Phoenix (KO 103,
KI 70, coupon barrier 80, 1% per period). Both run 1 year with 12 records at k/12, so the snapped contract days are
d_k = 30, 61, 91, 122, 152, 182, 213, 243, 274, 304, 335, 365. `start_date` = 2026-01-05, `initial_date` =
start - N, and the identity calendar is used. Spot is held at 100 (no KO, and the Phoenix coupon pays every
period). Full per-record rows are in `measure_clock_mismatch.json`. The snowball and the Phoenix give identical
timing; for the Phoenix, every coupon books on the same day as its record.

### 2.1 Where each record is booked and where the copy puts it

| N | record k booked on run day | copy holds record k through run day | copy's remaining time for k on its booking close | terminal settled on | last record booked on |
|---|---|---|---|---|---|
| 0 | d_k | d_k - 1 (dropped on its close) | 0 (dropped) | 365 | 365 |
| 30 | d_k - 30 (record 0 on day 0) | d_k - 1 | **30 days** (every record) | 365 | **335** |
| 100 | max(0, d_k - 100): records 0-2 all on day 0 | d_k - 1 | **min(d_k, 100) days** (30, 61, 91, then 100) | 365 | **265** |

Knock-out path (spot 104 every close, `ko_path` in the json):

| N | tracker knocks out on | record | ledger `determination_time` | copy's first KO record (day-0 copy) |
|---|---|---|---|---|
| 0 | day 30 | 0 | 30/365 | day 30 |
| 30 | **day 0** | 0 | **0.0** | day 30 |
| 100 | **day 0** | 0 (record 0 of three records already behind the run) | **0.0** | day 30 |

### 2.2 PV effect on the copy

The tracker's copy, which ages from `start_date`, is compared with the copy its own event clock implies, which ages
from `initial_date`. Both are priced with `SnowballPDESolver` / `PhoenixPDESolver` at stock `PDEParams()`. The
environment is spot 100, vol 20%, r 3%, notional 100. The table shows the difference (copy - event-clock copy) in bp
of notional; N = 0 is exactly 0.0 on every date.

| contract | N | day 0 | day 15 | day 45 | day 180 | day 300 |
|---|---|---|---|---|---|---|
| snowball | 30 | -124.7 | -125.1 | -129.7 | -124.5 | -46.2 |
| snowball | 100 | -412.1 | -430.8 | -441.6 | -372.3 | -63.1 |
| Phoenix | 30 | -18.2 | -18.2 | -19.0 | -11.9 | +5.4 |
| Phoenix | 100 | -68.3 | -65.9 | -66.1 | -13.7 | +21.2 |

Example: the snowball with N = 30 on day 0. The copy holds 12 records and 365 days, PV 2.3395. The event clock holds
11 records and 335 days, PV 3.5866.

### 2.3 Control: the same contract on dates

`dated_catch_up.py` runs the same snowball with its records on dates (`initial_date + d_k`) and
`exercise_date = initial_date + 365`. It has one clock, and its results are:

- N = 0: records booked on d_k, maturity on day 365.
- N = 30: records booked on d_k - 30, maturity on day **335**, day-0 copy holds 12 records.
- N = 100: records 0-2 booked on day 0, then d_k - 100, maturity on day **265**, day-0 copy holds 9 records.

The numeric contract's **event clock already matches its dated twin**. The parts that do not match the dated twin
are the copy, the terminal and the ledger times. Item 2 of section 1 (records dated before the run are tested
against the first close) is also the dated contract's behaviour. The tracker has no history filter for either kind
of contract; a caller that wants those records treated as history has to carry them in `observed_*_indices`.

## 3. Who hits it

All production entry points take the product as booked and set `start_date` to the run's first day:

| Caller | `start_date` | file |
|---|---|---|
| `PortfolioLifecycleManager.register_positions` | `base_date` | `asset/equity/lifecycle/manager.py`:131-134 |
| - equity backtest engine | `config.start_date` | `backtest/equity/engine.py`:205-206 |
| - dynamic scenario | `day_path.start_date`, else the first env's valuation date | `dynamicscenario/engine.py`:137-149 |
| - PnL-explain / portfolio tests and users of the manager | `base_date` | (same manager) |
| Replay `ProductReplay` (book backtest, OTC autocallable backtest) | first market-data date >= `config.start_date` | `backtest/replay/engine.py`:286-287, `product_replay.py`:181 |
| Simulated-path backtest (`_tracker`, repricing, surface) | `dates[0]` | `backtest/simulation/engine.py`:70/201/209/226, `pricing/repricing.py`:229/234 |
| Intraday provisional replay | `getattr(product, "initial_date")` | `intraday/provisional.py`:331-332 (**never hits**: start == initial) |

No library code sets `initial_date` on a numeric product; the value is whatever the booking carries. Production paths
hit the mismatch whenever a numeric contract carries `initial_date` and the run starts on another day:

- **A mid-life run of a booked contract.** Examples: a replay with `config.start_date` after inception, a dynamic
  scenario run from today, or an equity backtest or PnL-explain window that opens mid-life.
- **A gap at inception in the market data.** The replay starts on the first market date on or after
  `config.start_date`. If the inception day is missing from the data, or falls on a non-trading day, the run starts
  1+ days late and the mismatch appears without any warning.

The repository's own numeric-with-`initial_date` users all start at inception, so the gap is 0 there today. These
are `example/mo_volmodels/12_snowball_volmodel_backtest.py`:1035 (its comment reads "The lifecycle tracker resolves
the observation schedule off this date"), `example/snowball_volmodel_backtest/terms.py`:275,
`example/snowball_q_term_structure/_common.py`:873 and `test/simulation/conftest.py`:96. Each is safe only because
its market data contains the inception day.

**Test-suite scan.** The runtime plugin was loaded in logging-only mode (`TRACKER_CLOCK_OPTION=NONE`) over the 125
test files that mention lifecycle or replay (`runs/testlist.txt`; this excludes `mo_volmodels`, `modelvalidation`,
`volcalibration`, `fx_volmodels` and `pde_bitwise`). Result: **2514 passed, 48 skipped.** The condition was met in
exactly **2 tests**, both written by the KO-reset branch (`runs/hits_NONE.jsonl`):

- `test/lifecycle_ko_reset/test_review_fixes.py::test_a_mid_life_run_places_schedule_two_after_the_knock_in`
- `test/lifecycle_ko_reset/test_review_fixes.py::test_a_numeric_mid_life_rebased_copy_is_timed_like_its_events`

(N = 100, KO-reset REBASED.) No replay golden, simulation, dynamic-scenario, PnL-explain or plain snowball/Phoenix
test meets the condition.

Caveat: spawn-pool workers (the simulation runner with `workers > 1`) do not load the plugin, so they are not
logged. Every pooled test in the list also runs in-process, and none of those in-process runs hit the condition.

## 4. Options and what each moves

Each option was applied at runtime by `tracker_clock_options.py` (a pytest plugin, also imported by the measurement
script). Every arm changes the tracker **only** when the condition holds: numeric timing, both dates set, different
days. Outside the condition, every call runs the original code. A test that never meets the condition therefore
cannot change under any arm, which is why the arms were run only on `test/lifecycle_ko_reset/` (117 tests), the
directory that holds both hitting tests.

### A. Numeric record times are measured from `initial_date` when it is set

This makes one clock, `initial_date`, for everything: copy ageing, the valuation point, the terminal's due day,
ledger times, and a REBASED schedule 2's base and knock-in time. The runtime patch moves `start_date` to
`initial_date` for the duration of each public call.

- **Repro under A** (`measure_clock_mismatch_A.json`): the copy drops every record on the close that books it
  (lead -1, the same as N = 0). The terminal settles on day 335 (N = 30) and day 265 (N = 100), matching the last
  record. The day-0 copy holds 12 and 9 records. **These are the dated twin's numbers exactly** (section 2.3). A KO
  of record 0 books `determination_time` 30/365 (N = 30) and 100/365 (N = 100, the catch-up close).
- **Tests** (`runs/pytest_A.txt`): **117 passed, nothing changes.** Both hitting tests pass, because they assert
  event days and copy offsets that A reproduces: the knock-in falls on run day 60, schedule 2 hangs 30 days later,
  and the day-70 copy holds it 20 and 50 days ahead. No golden changes.
- **What moves in production:** any numeric contract with an earlier `initial_date`. Its copy ages N days further
  (PV moves by the section 2.2 amounts, towards the event-clock column), its terminal settles N days earlier, and its
  ledger times shift by N/365.
- **Implementation scope** (not done here): add one clock-origin accessor (`initial_date` for a numeric contract
  when set, else `start_date`). Then use it in `product_for_pricing`, `_valuation_point`, `_schedule_two_base` /
  `knock_in_time`, and the simulation's `elapsed_time` / `surface.elapsed`, which the runtime patch does not touch
  and no test exercises with a gap. Also correct the `_schedule_two_base` docstring, which names `start_date` as the
  copy's origin. Records dated before the run keep today's catch-up rule, the same as dated contracts (see the open
  point below).

### B. Numeric record times are measured from `start_date`

Under B, `initial_date` is never read for a numeric contract. Records, settlements and the terminal are dated from
the run start. The runtime patch makes `_record_dicts`, `_terminal_payment_date` and `_schedule_resolution_env` use
`start_date`.

- **Repro under B** (`measure_clock_mismatch_B.json`): every N gives the N = 0 table. Records are booked on d_k, the
  terminal on day 365, the copy is consistent, and the first KO is on day 30. **The contract is treated as freshly
  issued at the run start**, whatever its `initial_date` says.
- **Tests** (`runs/pytest_B.txt`): **2 failed, 115 passed.** Both hitting tests fail:
  `test_a_mid_life_run_places_schedule_two_after_the_knock_in` loses its knock-in (the KI record at contract day 160
  is now due on run day 160, not 60), and `test_a_numeric_mid_life_rebased_copy_is_timed_like_its_events` fails for the same reason: with no knock-in,
  the day-70 copy holds the unconverted REBASED offsets (30 and 60 days) instead of 20 and 50.
- **What moves in production:** the event days of every numeric contract with an earlier `initial_date`, which move
  N days later. The copy's PV does not change. This contradicts the stated intent of the repository's own callers,
  which set `initial_date` precisely to anchor the schedule. For a real mid-life booking it books every observation
  N days late.

### C. Refuse a numeric-schedule contract whose `initial_date` is not the run start

The patch raises `ValidationError` on any public tracker call under the condition.

- **Repro under C** (`measure_clock_mismatch_C.json`): N = 30 and N = 100 are refused, and N = 0 is unchanged.
- **Tests** (`runs/pytest_C.txt`): **2 failed, 115 passed.** Both hitting tests are refused.
- **What moves in production:** a mid-life numeric run, or one whose inception falls in a data gap, stops with an
  error instead of mispricing. The user must either start the run on `initial_date` or re-express the contract
  (dates, or age it to the run start and drop `initial_date`).

## 5. Recommendation

**A.** The reasons:

- It is the only option under which a numeric contract behaves exactly like its dated twin (section 2.3 against the
  A repro).
- It matches what the code and its callers already say `initial_date` means. `_schedule_resolution_env` is
  documented as resolving "from the contract issue date", and the examples set `initial_date` because "the lifecycle
  tracker resolves the observation schedule off this date".
- The event clock already uses `initial_date`. A only has to bring the copy, the terminal and the ledger onto it.
- It changes no existing test or golden.

B gets internal consistency by moving every event off the term sheet. C is the safe fallback if you would rather not
define the semantics now. If you choose A, it costs one accessor plus the two simulation `elapsed` helpers.

**Open point for a separate decision, shared with dated contracts:** records dated before `start_date` are booked
on the run's first close and tested against that close's spot. Under N = 30 with spot >= KO on day 0, the contract
knocks out on day 0 on a record that was contractually observed on `initial_date + 30`, which is the run start
itself, so that case is correct. Under N = 100, records 0-2 (days -70, -39, -9) are all tested against day 0's close.
A real mid-life run should carry those observations in `observed_*_indices` (or carried KI state), or the tracker
should refuse a run whose uncarried records lie before `start_date`. Option A does not change this. It applies
equally to dated contracts today.

## 6. Status: implemented (A), with decision 2

Branch `w1-tracker-clock` from main dcea77b6 (commits bf649ef9, e9990119, 93b8d63b and this note).

**One clock.** `lifecycle_clock_origin(product, start_date)` (`lifecycle/autocallable.py`) is the contract's
`initial_date` when set, else `start_date`. The tracker exposes it as `clock_origin`. It drives the record dates,
`_terminal_payment_date`, `_schedule_resolution_env`, the pricing copy's ageing (in whole days), `_valuation_point`
(the numeric terminal's due day and the ledger times), and the KO-reset `_schedule_two_base` / `knock_in_time`. The
`_schedule_two_base` docstring is corrected. Consumers that re-derive the clock use the same function:
`CalendarSchedule.clock_origin` -> `VectorLifecycle.elapsed_time`; `LifeSurfacePricer` (its elapsed time, its
day-0 node and its solve's valuation date); the manager's run ledger (each flow on its own tracker's origin); the
replay dashboard copy; and the PnL-explain recorder's valuation point (autocallable states only).

**Decision 2.** A KO / KI / coupon record dated before `start_date` and not carried as observed is marked observed
and never tested. KO and KI did not fire. A coupon there is history: no cash, and nothing is memorized. A record
dated on `start_date` is tested on the first close. `resolve_calendar_schedule` maps such a record to due day -1.
This applies to numeric and dated contracts alike. No warning is printed (ruling: every mid-life run has such
records, and the carried-state fields are the channel for a real pre-run event). Rulings 1-4 of the hand-back were
signed by the user on 2026-10-09.

**Matured before the run** (user ruling 2026-10-09: "print a maturity warning, and return zeroes"; commit 510bf6a7).
A contract whose maturity falls strictly before `start_date` is never settled in the run. The maturity date is the
terminal's due day under the decision-2 rule. For a snowball or Phoenix it is the terminal timing's determination date,
or on the numeric clock the origin plus round(t*365) days. For a KO-reset it is V0 at schedule 1's last record (R2),
or V1 at the knocked-in end once knocked in, carried or not. The tracker emits `MaturedBeforeRunWarning` (a
`UserWarning`) once per tracker, naming the contract, the maturity date and the run start. It then marks the state
dead, matured and settled, with no ledger entry. Every consumer values the contract at zero on every day:
- `PortfolioLifecycleManager.register_positions` drops the position before the first valuation.
- The replay prices the initial book value at 0 and never marks the contract.
- The ensemble flags it (`CalendarSchedule.matured_before_run`): `VectorLifecycle` starts it dead, and the day-0 gate
  and the initial book value skip it.

A maturity on `start_date` is still settled on the first close. Numeric and dated twins behave identically
(`test/test_matured_before_run.py`, 25 tests). After this change the same 229-file list gives 3702 passed and 61
skipped, with 0 failed. No existing expected value changed.

**Parity** (`test/test_tracker_clock_parity.py`): a snowball, a memory Phoenix and a KO-reset (ABSOLUTE and REBASED),
each written numerically (k/12 + `initial_date`) and on dates (`initial_date + round(k/12*365)`), are run with
N = 0, 30 and 100 on quiet, knock-out and knock-in paths. The event days, the ledger cash and the determination and
payment instants are identical. The copy PVs (PDE) agree to 1.0e-12 per 100 notional at worst, the ulp-level
difference between a snapped time aged by days and a day count of two dates. Each contract day prices the same at
every N. Cash that accrues cannot agree across the forms: a numeric record accrues k/12 and a dated one d_k/365, so
an annualized 15% snowball's record 0 pays 1.25 against 1.2329 already at N = 0. The annualized variants are
therefore held to identical timing, and to cash that does not depend on N within each form.

**What moved.** Test list: the findings' 125 files, plus every test file that mentions
lifecycle/replay/backtest/simulation/dynamicscenario/pnlexplain/intraday/tracker (227 files, excluding mo_volmodels,
modelvalidation, volcalibration, fx_volmodels and pde_bitwise). Baseline dcea77b6: 3561 passed, 61 skipped. The
branch: 3677 passed (the 116 new tests), 61 skipped, 0 failed. **No existing expected value changed.** The
simulation surface on a contract issued 2 days early: the readout against exact repricing was 268.5 bp off on the
old clock and is 0.40 bp off now.

**Pre-existing, not changed here.** (1) The replay dashboard ages the unsnapped booked contract, so an off-day
numeric time sits up to half a day from the tracker copy's. (2) A numeric delayed maturity never sets the state's
scalar `settlement_date`, so a walk that settles by date disagrees with `VectorLifecycle` once the maturity's payment
day falls inside the calendar. This happens at N = 0 as well.
